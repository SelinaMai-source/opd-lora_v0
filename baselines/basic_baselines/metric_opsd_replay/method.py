from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from baselines.basic_baselines.ce_opsd.method import _encode_supervised, _example_golds, _has_supervised
from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod
from baselines.basic_baselines.opsd.method import pad_right, pad_right_labels, token_mean_logprob
from core.data import Example
from core.evaluate import _score_task_aware
from core.formatting import format_for_infer
from core.metrics_utils import rouge_l_fscore, sentence_bleu4

_LONG_SEQ_SKIP = 1024
_CLOSED_TYPES = {"label_accuracy", "after_step_extracted_em"}
_MIN_GAP = 1e-4


class MetricOpsdReplayMethod(CeOpsdReplayMethod):
    """Proto-replay CE+OPSD plus gated open-ended ranking (v3 metric gate).

    Every ``every_n_batches`` (default 8): sample 4 candidates. A pair enters
    the extra loss only if the winner is ≥ the loser on **both** BLEU and
    ROUGE-L, the gap is clear, texts differ, and the task is open-ended.
    Pairwise uses length-normalized ``token_mean_logprob``. Closed / invalid
    cases stay on CE+OPSD only.
    """

    name = "metric_opsd_replay"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        metric_cfg = cfg.get("metric_opsd", {}) if isinstance(cfg.get("metric_opsd", {}), dict) else {}
        self.mu = float(metric_cfg.get("mu", 0.02))
        self.every_n_batches = int(metric_cfg.get("every_n_batches", 8))
        self.n_candidates = int(metric_cfg.get("n_candidates", 4))
        self.pref_beta = float(metric_cfg.get("pref_beta", metric_cfg.get("beta", 1.0)))
        self.gated_pairs = bool(metric_cfg.get("gated_pairs", True))
        self._rank_vals: List[float] = []
        self._n_rank = 0
        self._n_gate_skip = 0
        self._n_closed_skip = 0
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
        self._n_gate_skip = 0
        self._n_closed_skip = 0
        self._n_skip = 0
        metrics = super().train_on_segment(
            segment=segment, model=model, lora=lora, lr=lr, epochs=epochs, batch_size=batch_size
        )
        metrics["mu"] = float(self.mu)
        metrics["metric_every_n"] = int(self.every_n_batches)
        metrics["n_candidates"] = int(self.n_candidates)
        metrics["gated_pairs"] = bool(self.gated_pairs)
        metrics["train.rank"] = sum(self._rank_vals) / max(1, len(self._rank_vals))
        metrics["n_rank_batches"] = int(self._n_rank)
        metrics["n_gate_skip"] = int(self._n_gate_skip)
        metrics["n_closed_skip"] = int(self._n_closed_skip)
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        losses = []
        for row in ce_rows:
            ex: Example = row["ex"]
            golds = list(row.get("golds") or _example_golds(ex))
            enc = row["enc"]
            if bool(row.get("long")) or len(enc.full_ids) >= _LONG_SEQ_SKIP:
                self._n_skip += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            gold = str(golds[0] if golds else "")
            score_type = _task_score_type(ex, gold)
            if score_type in _CLOSED_TYPES:
                self._n_closed_skip += 1
                continue
            pref = _open_gated_rank_pref(
                model=model,
                tokenizer=tokenizer,
                device=device,
                pad_id=pad_id,
                max_len=max_len,
                max_new=max_new,
                temperature=self.rollout_temperature,
                top_p=self.rollout_top_p,
                ex=ex,
                golds=golds,
                n_candidates=self.n_candidates,
                beta=self.pref_beta,
                gated=self.gated_pairs,
            )
            if pref is None:
                self._n_gate_skip += 1
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


def _best_bleu_rouge(pred: str, golds: Sequence[str]) -> Tuple[float, float]:
    refs = [str(g) for g in golds if str(g)] or [""]
    best_bleu = -1.0
    best_rouge = -1.0
    best_sum = -1.0
    for gold in refs:
        bleu = float(sentence_bleu4(pred, gold))
        rouge = float(rouge_l_fscore(pred, gold))
        total = bleu + rouge
        if total > best_sum:
            best_sum = total
            best_bleu = bleu
            best_rouge = rouge
    return best_bleu, best_rouge


def _mean_token_logp(model: Any, tokenizer: Any, device: Any, pad_id: int, max_len: int, ex: Example, target: str):
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    enc = _encode_supervised(model, tokenizer, ex.instruction, ex.input, target, max_len)
    if not _has_supervised(enc.labels):
        return None
    ids, mask = pad_right([enc.full_ids], pad_id=pad_id, device=device)
    labels = pad_right_labels([enc.labels], device=device)
    logits = model.forward_logits(ids, mask)
    logp = token_mean_logprob(logits, labels)
    out = logp.reshape(-1)[0]
    del logits, ids, mask, labels, logp
    return out


def _open_gated_rank_pref(
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
    golds: Sequence[str],
    n_candidates: int,
    beta: float,
    gated: bool,
) -> Optional[Any]:
    import torch
    import torch.nn.functional as F

    prompt = format_for_infer(tokenizer, ex.instruction, ex.input)
    n = max(2, int(n_candidates))
    scored: List[Tuple[float, float, str]] = []
    for _ in range(n):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        one = model.sample_rollouts(
            [prompt],
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
        )
        pred = tokenizer.decode(one[0], skip_special_tokens=True) if one and one[0] else ""
        if not pred.strip():
            continue
        bleu, rouge = _best_bleu_rouge(pred, golds)
        scored.append((bleu, rouge, pred))
    if len(scored) < 2:
        return None
    # Winner = max BLEU+ROUGE; loser = min. Gate requires winner ≥ loser on both.
    scored.sort(key=lambda x: (x[0] + x[1], x[0], x[1]), reverse=True)
    high_bleu, high_rouge, high_txt = scored[0]
    low_bleu, low_rouge, low_txt = scored[-1]
    if high_txt.strip() == low_txt.strip():
        return None
    if gated:
        if high_bleu + 1e-12 < low_bleu or high_rouge + 1e-12 < low_rouge:
            return None
        if (high_bleu - low_bleu) + (high_rouge - low_rouge) < _MIN_GAP:
            return None
    elif high_bleu + high_rouge <= low_bleu + low_rouge:
        return None
    high_lp = _mean_token_logp(model, tokenizer, device, pad_id, max_len, ex, high_txt)
    low_lp = _mean_token_logp(model, tokenizer, device, pad_id, max_len, ex, low_txt)
    if high_lp is None or low_lp is None:
        return None
    return -F.logsigmoid(float(beta) * (high_lp - low_lp))
