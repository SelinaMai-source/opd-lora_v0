from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from baselines.basic_baselines.ce_opsd.method import (
    CeOpsdMethod,
    _encode_supervised,
    _example_golds,
    _has_supervised,
)
from baselines.basic_baselines.neg_opsd.method import DEFAULT_NEG_TEMPLATES
from baselines.basic_baselines.opsd.method import causal_lm_ce, pad_right, pad_right_labels
from core.data import Example
from core.evaluate import _score_task_aware
from core.formatting import format_for_infer
from core.metrics_utils import rouge_l_fscore, sentence_bleu4, token_f1

_LONG_SEQ_SKIP = 1024
_CLOSED_TYPES = {"label_accuracy", "after_step_extracted_em"}


class MetricOpsdMethod(CeOpsdMethod):
    """
    CE+λ OPSD backbone plus metric ranking (open-ended) or format negatives (closed).

    Every ``every_n_batches`` (default 6): sample 4 candidates, score Token-F1 +
    ROUGE-L + BLEU vs the CE gold, pairwise push high / press low. Closed-form
    tasks use neg_opsd templates + exact-match instead of ranking.
    """

    name = "metric_opsd"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        metric_cfg = cfg.get("metric_opsd", {}) if isinstance(cfg.get("metric_opsd", {}), dict) else {}
        self.mu = float(metric_cfg.get("mu", 0.1))
        self.every_n_batches = int(metric_cfg.get("every_n_batches", 6))
        self.n_candidates = int(metric_cfg.get("n_candidates", 4))
        self.pref_beta = float(metric_cfg.get("pref_beta", metric_cfg.get("beta", 1.0)))
        raw_templates = metric_cfg.get("templates", DEFAULT_NEG_TEMPLATES)
        if not isinstance(raw_templates, list) or not raw_templates:
            raw_templates = DEFAULT_NEG_TEMPLATES
        self.templates = [str(t) for t in raw_templates]
        self._rank_vals: List[float] = []
        self._n_rank = 0
        self._n_closed = 0
        self._n_skip = 0

    def train_on_segment(
        self,
        *,
        segment,
        model: Any,
        lora: Any,
        lr: float,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        self._rank_vals = []
        self._n_rank = 0
        self._n_closed = 0
        self._n_skip = 0
        metrics = super().train_on_segment(
            segment=segment, model=model, lora=lora, lr=lr, epochs=epochs, batch_size=batch_size
        )
        metrics["mu"] = float(self.mu)
        metrics["metric_every_n"] = int(self.every_n_batches)
        metrics["n_candidates"] = int(self.n_candidates)
        metrics["train.rank"] = sum(self._rank_vals) / max(1, len(self._rank_vals))
        metrics["n_rank_batches"] = int(self._n_rank)
        metrics["n_closed_pref"] = int(self._n_closed)
        metrics["n_metric_skip"] = int(self._n_skip)
        return metrics

    def _maybe_extra_loss(
        self,
        *,
        batch_idx: int,
        ce_rows: List[Dict[str, Any]],
        model: Any,
        lora: Any,
        tokenizer: Any,
        device: Any,
        max_len: int,
        pad_id: int,
        max_new: int,
        **_kwargs: Any,
    ):
        if self.mu <= 0 or self.every_n_batches <= 0:
            return None, {}
        if (batch_idx + 1) % self.every_n_batches != 0:
            return None, {}
        import torch
        import torch.nn.functional as F

        losses = []
        for row in ce_rows:
            ex: Example = row["ex"]
            gold = str(row.get("teacher_gold") or random.choice(_example_golds(ex)))
            enc = row["enc"]
            if len(enc.full_ids) >= _LONG_SEQ_SKIP:
                self._n_skip += 1
                continue
            score_type = _task_score_type(ex, gold)
            if score_type in _CLOSED_TYPES:
                pref = _closed_format_pref(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    pad_id=pad_id,
                    max_len=max_len,
                    ex=ex,
                    gold=gold,
                    templates=self.templates,
                    beta=self.pref_beta,
                )
                if pref is None:
                    self._n_skip += 1
                    continue
                losses.append(pref)
                self._n_closed += 1
                continue
            pref = _open_rank_pref(
                model=model,
                tokenizer=tokenizer,
                device=device,
                pad_id=pad_id,
                max_len=max_len,
                max_new=max_new,
                temperature=self.rollout_temperature,
                top_p=self.rollout_top_p,
                ex=ex,
                gold=gold,
                n_candidates=self.n_candidates,
                beta=self.pref_beta,
            )
            if pref is None:
                self._n_skip += 1
                continue
            losses.append(pref)
            self._n_rank += 1
        if not losses:
            return None, {}
        extra = torch.stack(losses).mean() * float(self.mu)
        val = float(extra.detach().item())
        self._rank_vals.append(val)
        return extra, {"train.rank_acc": val}


def _task_score_type(ex: Example, gold: str) -> str:
    info = _score_task_aware(
        pred=gold,
        gold=gold,
        norm_pred=gold,
        norm_gold=gold,
        instruction=ex.instruction,
        input_text=ex.input,
        cfg={},
    )
    return str(info.get("task_score_type", "strict_em"))


def _candidate_score(pred: str, gold: str) -> float:
    return float(token_f1(pred, gold) + rouge_l_fscore(pred, gold) + sentence_bleu4(pred, gold))


def _mean_token_logp(model: Any, tokenizer: Any, device: Any, pad_id: int, max_len: int, ex: Example, target: str):
    import torch

    enc = _encode_supervised(model, tokenizer, ex.instruction, ex.input, target, max_len)
    if not _has_supervised(enc.labels):
        return None
    ids, mask = pad_right([enc.full_ids], pad_id=pad_id, device=device)
    labels = pad_right_labels([enc.labels], device=device)
    logits = model.forward_logits(ids, mask)
    nll = causal_lm_ce(logits, labels)
    logp = -nll
    del logits, ids, mask, labels, nll
    return logp


def _closed_format_pref(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    pad_id: int,
    max_len: int,
    ex: Example,
    gold: str,
    templates: List[str],
    beta: float,
):
    import torch.nn.functional as F

    gold_lp = _mean_token_logp(model, tokenizer, device, pad_id, max_len, ex, gold)
    if gold_lp is None:
        return None
    neg_lps = []
    for tmpl in templates:
        neg = str(tmpl).replace("{gold}", gold)
        if neg.strip() == gold.strip():
            continue
        lp = _mean_token_logp(model, tokenizer, device, pad_id, max_len, ex, neg)
        if lp is not None:
            neg_lps.append(lp)
    if not neg_lps:
        return None
    import torch

    logp_neg = torch.stack(neg_lps)
    return -F.logsigmoid(float(beta) * (gold_lp - logp_neg)).mean()


def _open_rank_pref(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    pad_id: int,
    max_len: int,
    max_new: int,
    temperature: float,
    top_p: float,
    ex: Example,
    gold: str,
    n_candidates: int,
    beta: float,
):
    import torch.nn.functional as F

    prompt = format_for_infer(tokenizer, ex.instruction, ex.input)
    prompts = [prompt] * max(2, n_candidates)
    rollouts = model.sample_rollouts(
        prompts,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
    )
    scored: List[Tuple[float, str]] = []
    for ro in rollouts:
        pred = tokenizer.decode(ro, skip_special_tokens=True) if ro else ""
        scored.append((_candidate_score(pred, gold), pred))
    if len(scored) < 2:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    high_score, high_txt = scored[0]
    low_score, low_txt = scored[-1]
    if high_txt.strip() == low_txt.strip() or high_score <= low_score:
        return None
    high_lp = _mean_token_logp(model, tokenizer, device, pad_id, max_len, ex, high_txt)
    low_lp = _mean_token_logp(model, tokenizer, device, pad_id, max_len, ex, low_txt)
    if high_lp is None or low_lp is None:
        return None
    return -F.logsigmoid(float(beta) * (high_lp - low_lp))
