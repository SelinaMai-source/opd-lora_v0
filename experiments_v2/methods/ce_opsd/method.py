from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

from baselines.basic_baselines.opsd.method import (
    causal_lm_ce,
    pad_right,
    pad_right_labels,
    run_opsd_step,
)
from core.data import Example, Segment
from core.formatting import format_for_infer
from core.metrics_utils import rouge_l_fscore, token_f1
from core.train_labels import build_supervised_labels

# CNN/DM-scale sequences already fill ~47GiB after one forward; skip the extra
# OPSD student/teacher forwards (same cutoff as neg_opsd). Keep CE only.
_LONG_SEQ_SKIP_OPSD = 1024


class CeOpsdMethod:
    """
    Same-step CE + on-policy OPSD: L = L_CE + λ L_OPSD.

    - CE: randomly pick one gold from ``ex.outputs`` (the only gold if singleton).
    - OPSD: student rollout on current instruction+input only; teacher gold is the
      reference with highest Token F1 vs the rollout (ROUGE-L tie-break).
    - Not v1 ``gold_opsd`` (off-policy KL on gold tokens).
    """

    name = "ce_opsd"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        opsd_cfg = cfg.get("opsd", {}) if isinstance(cfg.get("opsd", {}), dict) else {}
        ce_cfg = cfg.get("ce_opsd", {}) if isinstance(cfg.get("ce_opsd", {}), dict) else {}
        self.lambda_opsd = float(ce_cfg.get("lambda_opsd", 0.2))
        self.divergence = str(opsd_cfg.get("divergence", "jsd")).strip().lower()
        if self.divergence not in {"jsd", "kl"}:
            raise ValueError(f"opsd.divergence must be 'jsd' or 'kl', got: {self.divergence}")
        self.beta = float(opsd_cfg.get("beta", 0.5))
        self.rollout_temperature = float(opsd_cfg.get("rollout_temperature", 1.0))
        self.rollout_top_p = float(opsd_cfg.get("rollout_top_p", 1.0))
        self.rollout_max_new_tokens = int(opsd_cfg.get("rollout_max_new_tokens", 0))
        self.teacher_format_constraint = bool(opsd_cfg.get("teacher_format_constraint", True))

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {"active_adapter": lora.get_active_adapter_name()}

    def _training_examples(self, *, segment: Segment, model: Any, lora: Any) -> List[Example]:
        return list(segment.train)

    def _teacher_prototypes_for(self, ex: Example) -> Optional[List[Dict[str, Any]]]:
        return None

    def _maybe_extra_loss(self, **_kwargs: Any):
        return None, {}

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

        tokenizer = model.tokenizer
        device = model.device
        max_len = int(model.cfg.max_seq_len)
        pad_id = int(tokenizer.pad_token_id)
        max_new = self.rollout_max_new_tokens or int(model.cfg.gen_max_new_tokens)

        metrics: Dict[str, Any] = {"batches": 0}
        ce_vals: List[float] = []
        opsd_vals: List[float] = []
        loss_vals: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        rollout_token_total = 0
        rollout_count = 0
        skipped_long = 0

        train_examples = self._training_examples(segment=segment, model=model, lora=lora)
        batch_idx = 0
        for _ in range(max(1, epochs)):
            for batch in _batch_examples(train_examples, batch_size):
                ce_rows: List[Dict[str, Any]] = []
                for ex in batch:
                    golds = _example_golds(ex)
                    ce_gold = random.choice(golds)
                    enc = _encode_supervised(model, tokenizer, ex.instruction, ex.input, ce_gold, max_len)
                    if not _has_supervised(enc.labels):
                        continue
                    ce_rows.append(
                        {
                            "ex": ex,
                            "golds": golds,
                            "enc": enc,
                            "long": len(enc.full_ids) >= _LONG_SEQ_SKIP_OPSD,
                        }
                    )
                if not ce_rows:
                    continue

                opsd_rows = [row for row in ce_rows if not row["long"]]
                skipped_long += len(ce_rows) - len(opsd_rows)
                if any(row["long"] for row in ce_rows):
                    torch.cuda.empty_cache()

                rollouts: List[List[int]] = []
                if opsd_rows:
                    student_prompts = [
                        format_for_infer(tokenizer, row["ex"].instruction, row["ex"].input)
                        for row in opsd_rows
                    ]
                    rollouts = model.sample_rollouts(
                        student_prompts,
                        max_new_tokens=max_new,
                        temperature=self.rollout_temperature,
                        top_p=self.rollout_top_p,
                    )
                    for row, ro in zip(opsd_rows, rollouts):
                        pred = tokenizer.decode(ro, skip_special_tokens=True) if ro else ""
                        row["teacher_gold"] = _pick_nearest_gold(pred, row["golds"])

                if hasattr(model, "model") and hasattr(model.model, "train"):
                    model.model.train()

                s_ids_list = [row["enc"].full_ids for row in ce_rows]
                s_labels_list = [row["enc"].labels for row in ce_rows]
                s_ids, s_mask = pad_right(s_ids_list, pad_id=pad_id, device=device)
                s_labels = pad_right_labels(s_labels_list, device=device)
                student_logits = model.forward_logits(s_ids, s_mask)
                ce = causal_lm_ce(student_logits, s_labels)

                opsd_loss = None
                opsd_val = 0.0
                if opsd_rows:
                    b_pairs = [(row["ex"].instruction, row["ex"].input) for row in opsd_rows]
                    b_targets = [str(row["teacher_gold"]) for row in opsd_rows]
                    teacher_protos = [
                        self._teacher_prototypes_for(row["ex"]) for row in opsd_rows
                    ]
                    step = run_opsd_step(
                        model=model,
                        lora=lora,
                        b_pairs=b_pairs,
                        b_targets=b_targets,
                        lr=lr,
                        divergence=self.divergence,
                        beta=self.beta,
                        rollout_temperature=self.rollout_temperature,
                        rollout_top_p=self.rollout_top_p,
                        rollout_max_new_tokens=self.rollout_max_new_tokens,
                        teacher_source="base",
                        teacher_format_constraint=self.teacher_format_constraint,
                        rollouts=rollouts,
                        do_step=False,
                        teacher_prototypes=teacher_protos,
                    )
                    if not step.get("skipped"):
                        opsd_loss = step["loss_tensor"]
                        opsd_val = float(step["loss"])
                        rollout_token_total += int(step["rollout_token_total"])
                        rollout_count += int(step["rollout_count"])

                loss = ce if opsd_loss is None else ce + float(self.lambda_opsd) * opsd_loss
                extra_loss, extra_stats = self._maybe_extra_loss(
                    batch_idx=batch_idx,
                    ce_rows=ce_rows,
                    model=model,
                    lora=lora,
                    tokenizer=tokenizer,
                    device=device,
                    max_len=max_len,
                    pad_id=pad_id,
                    max_new=max_new,
                )
                if extra_loss is not None:
                    loss = loss + extra_loss
                    for k, v in extra_stats.items():
                        metrics[k] = metrics.get(k, 0.0) + float(v)
                if bool(torch.isnan(loss).item()) or bool(torch.isinf(loss).item()):
                    raise RuntimeError(f"ce_opsd loss is NaN/Inf: {float(loss.item())}")

                model._last_lr = float(lr)
                loss.backward()
                step_stats = lora.step_adapter()

                ce_vals.append(float(ce.detach().item()))
                opsd_vals.append(opsd_val)
                loss_vals.append(float(loss.detach().item()))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                metrics["batches"] += 1
                batch_idx += 1
                del student_logits, s_ids, s_mask, s_labels, ce, opsd_loss, loss, extra_loss

        metrics["train.loss"] = sum(loss_vals) / max(1, len(loss_vals))
        metrics["train.ce"] = sum(ce_vals) / max(1, len(ce_vals))
        metrics["train.opsd"] = sum(opsd_vals) / max(1, len(opsd_vals))
        metrics["lambda_opsd"] = float(self.lambda_opsd)
        metrics["teacher_format_constraint"] = bool(self.teacher_format_constraint)
        metrics["opsd_skipped_long"] = int(skipped_long)
        metrics["rollout_mean_tokens"] = rollout_token_total / max(1, rollout_count)
        metrics["divergence"] = self.divergence
        metrics["beta"] = float(self.beta)
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["train.n_steps"] = int(metrics.get("batches", 0))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics


def _example_golds(ex: Example) -> List[str]:
    golds = [str(x) for x in (ex.outputs or []) if str(x)]
    if not golds:
        golds = [str(ex.output if ex.output is not None else "")]
    return golds


def _pick_nearest_gold(pred: str, golds: Sequence[str]) -> str:
    """Highest Token F1 vs rollout; ROUGE-L breaks ties; first gold wins remaining ties."""
    if len(golds) == 1:
        return golds[0]
    best_gold = golds[0]
    best_f1 = token_f1(pred, golds[0])
    best_rouge = rouge_l_fscore(pred, golds[0])
    for gold in golds[1:]:
        f1 = token_f1(pred, gold)
        if f1 > best_f1:
            best_gold = gold
            best_f1 = f1
            best_rouge = rouge_l_fscore(pred, gold)
        elif f1 == best_f1:
            rouge = rouge_l_fscore(pred, gold)
            if rouge > best_rouge:
                best_gold = gold
                best_rouge = rouge
    return best_gold


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


def _has_supervised(labels: List[int]) -> bool:
    return any(int(v) != -100 for v in labels)


def _batch_examples(examples: List[Example], batch_size: int):
    bs = max(1, int(batch_size))
    for i in range(0, len(examples), bs):
        yield examples[i : i + bs]
