from __future__ import annotations

from typing import Any, Dict, List

from baselines.basic_baselines.opsd.method import (
    _batch,
    _to_pairs,
    causal_lm_ce,
    pad_right,
    pad_right_labels,
)
from core.data import Segment
from core.train_labels import build_supervised_labels

DEFAULT_NEG_TEMPLATES = [
    "The answer is {gold}",
    "The final answer is {gold}",
    "Sure, {gold}",
]
# CNN/DM-scale sequences already fill ~47GiB after one forward; skip the 3
# extra neg forwards + pref logp (vocab logsumexp OOM at seg 27).
_LONG_SEQ_SKIP_PREF = 1024


class NegOPSDMethod:
    """
    Format pairwise preference on template negatives, stacked on SFT CE.

    L = L_CE + μ L_pref,  L_pref = -log σ(β (logp_gold − logp_neg))
    logp is length-normalized mean token log-prob on the supervised span.
    """

    name = "neg_opsd"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        neg_cfg = cfg.get("neg_opsd", {}) if isinstance(cfg.get("neg_opsd", {}), dict) else {}
        self.mu = float(neg_cfg.get("mu", 0.3))
        self.beta = float(neg_cfg.get("beta", 1.0))
        raw_templates = neg_cfg.get("templates", DEFAULT_NEG_TEMPLATES)
        if not isinstance(raw_templates, list) or not raw_templates:
            raw_templates = DEFAULT_NEG_TEMPLATES
        self.templates = [str(t) for t in raw_templates]

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {"active_adapter": lora.get_active_adapter_name()}

    def train_on_segment(
        self,
        *,
        segment: Segment,
        model: Any,
        lora: Any,
        lr: float,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F

        pairs, targets = _to_pairs(segment.train)
        tokenizer = model.tokenizer
        device = model.device
        max_len = int(model.cfg.max_seq_len)
        pad_id = int(tokenizer.pad_token_id)
        n_neg = len(self.templates)

        metrics: Dict[str, Any] = {"batches": 0}
        ce_vals: List[float] = []
        pref_vals: List[float] = []
        loss_vals: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                # One example at a time: packed gold+negs is (1+n_neg)*batch_size sequences
                # and OOMs Llama-8B at seq=2048. Accumulate grads, one optimizer step.
                groups: List[tuple] = []
                for (ins, inp), tgt in zip(b_pairs, b_targets):
                    gold_enc = _encode_supervised(model, tokenizer, ins, inp, str(tgt), max_len)
                    if not first_has_supervised(gold_enc.labels):
                        continue
                    ids_g = [gold_enc.full_ids]
                    labels_g = [gold_enc.labels]
                    for tmpl in self.templates:
                        neg = str(tmpl).replace("{gold}", str(tgt))
                        neg_enc = _encode_supervised(model, tokenizer, ins, inp, neg, max_len)
                        ids_g.append(neg_enc.full_ids)
                        labels_g.append(neg_enc.labels)
                    groups.append((ids_g, labels_g))

                kept = len(groups)
                if kept == 0:
                    continue

                ce_acc = 0.0
                pref_acc = 0.0
                loss_acc = 0.0
                for ids_g, labels_g in groups:
                    # One sequence at a time. Mean token logp = -CE (B=1 fused
                    # kernel) so we never materialize a float32 [T, vocab]
                    # logsumexp table. Ultra-long gold: CE only, skip pairwise.
                    gold_len = len(ids_g[0])
                    skip_pref = gold_len >= _LONG_SEQ_SKIP_PREF
                    if skip_pref:
                        torch.cuda.empty_cache()
                    seq_logps = []
                    ce = None
                    seq_iter = zip(ids_g[:1], labels_g[:1]) if skip_pref else zip(ids_g, labels_g)
                    for j, (seq_ids, seq_labs) in enumerate(seq_iter):
                        ids, mask = pad_right([seq_ids], pad_id=pad_id, device=device)
                        labels = pad_right_labels([seq_labs], device=device)
                        logits = model.forward_logits(ids, mask)
                        nll = causal_lm_ce(logits, labels)
                        if j == 0:
                            ce = nll
                        if not skip_pref:
                            seq_logps.append(-nll)
                        del logits, ids, mask, labels, nll
                    if skip_pref:
                        pref = ce * 0.0
                    else:
                        seq_logp = torch.stack(seq_logps)
                        logp_gold = seq_logp[0]
                        logp_neg = seq_logp[1:]
                        pref = -F.logsigmoid(float(self.beta) * (logp_gold - logp_neg)).mean()
                    loss = (ce + float(self.mu) * pref) / float(kept)
                    if bool(torch.isnan(loss).item()) or bool(torch.isinf(loss).item()):
                        raise RuntimeError(f"neg_opsd loss is NaN/Inf: {float(loss.item())}")
                    loss.backward()
                    ce_acc += float(ce.detach().item())
                    pref_acc += float(pref.detach().item())
                    loss_acc += float((ce + float(self.mu) * pref).detach().item())
                    del ce, pref, loss, seq_logps

                model._last_lr = float(lr)
                step_stats = lora.step_adapter()
                ce_mean = ce_acc / float(kept)
                pref_mean = pref_acc / float(kept)
                loss_mean = loss_acc / float(kept)

                ce_vals.append(ce_mean)
                pref_vals.append(pref_mean)
                loss_vals.append(loss_mean)
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                metrics["batches"] += 1

        metrics["train.loss"] = sum(loss_vals) / max(1, len(loss_vals))
        metrics["train.ce"] = sum(ce_vals) / max(1, len(ce_vals))
        metrics["train.pref"] = sum(pref_vals) / max(1, len(pref_vals))
        metrics["mu"] = float(self.mu)
        metrics["pref_beta"] = float(self.beta)
        metrics["n_neg_templates"] = int(n_neg)
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics


def _encode_supervised(model: Any, tokenizer: Any, ins: str, inp: str, target: str, max_len: int):
    return build_supervised_labels(
        tokenizer,
        ins,
        inp,
        str(target),
        max_len=max_len,
        min_target_tokens=int(model.cfg.min_target_tokens_for_loss),
        mask_eos_token_in_labels=bool(model.cfg.mask_eos_token_in_labels),
        mask_all_special_tokens_in_labels=bool(model.cfg.mask_all_special_tokens_in_labels),
        labeling_mode=str(model.cfg.train_labeling_mode),
        completion_only_response_template=str(model.cfg.completion_only_response_template),
    )


def first_has_supervised(labels: List[int]) -> bool:
    return any(int(v) != -100 for v in labels)
