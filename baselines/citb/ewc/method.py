from __future__ import annotations

import random
from typing import Any, Dict, List

import torch

from core.data import Segment

from baselines.citb.common import (
    batch_pairs,
    cfg_seed,
    empty_cuda_cache,
    ensure_default_adapter,
    iter_trainable_lora_params,
    reg_lambda,
    replay_k,
    sample_k,
    snapshot_trainable,
    summarize_sft_metrics,
    to_pairs,
    zero_lora_grad,
)


class CITBEWCMethod:
    """Segment-end Fisher on K memory examples; next segment L += 0.01 * Σ F (θ-θ*)²."""

    name = "citb_ewc"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.reg = reg_lambda(cfg, default=0.01)
        self.k = replay_k(cfg, default=50)
        self._rng = random.Random(cfg_seed(cfg))
        self._optpar: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}
        self._first_task = True

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "ewc_reg": self.reg,
            "ewc_k": self.k,
            "ewc_has_fisher": bool(self._fisher),
        }

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
        pairs, targets = to_pairs(segment.train)
        batch_accs: List[float] = []
        batch_losses: List[float] = []
        answer_token_accs: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        total_tokens = 0
        supervised_tokens = 0
        batches = 0
        use_reg = bool(self._fisher) and bool(self._optpar) and not self._first_task
        model._last_lr = float(lr)

        for _ in range(max(1, int(epochs))):
            idxs = list(range(len(pairs)))
            if len(idxs) > 1:
                self._rng.shuffle(idxs)
            order_pairs = [pairs[i] for i in idxs]
            order_targets = [targets[i] for i in idxs]
            for b_pairs, b_targets in batch_pairs(order_pairs, order_targets, batch_size):
                zero_lora_grad(lora)
                ce_loss, stats = model.compute_supervised_loss(b_pairs, b_targets)
                loss = ce_loss
                if use_reg:
                    loss = ce_loss + self._ewc_penalty(lora)
                loss.backward()
                step_stats = lora.step_adapter()
                batch_accs.append(float(stats.get("train_batch_acc", 0.0)))
                batch_losses.append(float(stats.get("train_loss", 0.0)))
                answer_token_accs.append(float(stats.get("train_answer_token_acc", 0.0)))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                total_tokens += int(stats.get("num_total_tokens", 0))
                supervised_tokens += int(stats.get("num_supervised_tokens", 0))
                batches += 1

        fisher_batches = self._refresh_fisher(model, lora, pairs, targets, batch_size)
        self._first_task = False
        empty_cuda_cache()
        metrics = summarize_sft_metrics(
            batches=batches,
            batch_accs=batch_accs,
            batch_losses=batch_losses,
            answer_token_accs=answer_token_accs,
            grad_norms=grad_norms,
            lr_values=lr_values,
            total_tokens=total_tokens,
            supervised_tokens=supervised_tokens,
            extra={
                "ewc_reg": self.reg,
                "ewc_k": self.k,
                "ewc_applied": bool(use_reg),
                "ewc_fisher_batches": int(fisher_batches),
            },
        )
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics

    def _ewc_penalty(self, lora: Any) -> torch.Tensor:
        penalty = None
        for name, param in iter_trainable_lora_params(lora):
            anchor = self._optpar.get(name)
            fisher = self._fisher.get(name)
            if anchor is None or fisher is None:
                continue
            term = (
                fisher.to(device=param.device, dtype=torch.float32)
                * (param.float() - anchor.to(device=param.device, dtype=torch.float32)).pow(2)
            ).sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            device = next((p.device for _, p in iter_trainable_lora_params(lora)), torch.device("cpu"))
            return torch.zeros((), device=device)
        return self.reg * penalty

    def _refresh_fisher(
        self,
        model: Any,
        lora: Any,
        pairs: List,
        targets: List[str],
        batch_size: int,
    ) -> int:
        mem_pairs, mem_targets = sample_k(pairs, targets, self.k, self._rng)
        self._optpar = snapshot_trainable(lora)
        fisher: Dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.detach().float().cpu())
            for name, param in iter_trainable_lora_params(lora)
        }
        n_batches = 0
        for b_pairs, b_targets in batch_pairs(mem_pairs, mem_targets, batch_size):
            if not b_pairs:
                continue
            zero_lora_grad(lora)
            loss, _stats = model.compute_supervised_loss(b_pairs, b_targets)
            loss.backward()
            for name, param in iter_trainable_lora_params(lora):
                if name not in fisher or param.grad is None:
                    continue
                fisher[name] += param.grad.detach().float().cpu().pow(2)
            n_batches += 1
        if n_batches > 0:
            for name in list(fisher.keys()):
                fisher[name] /= float(n_batches)
        zero_lora_grad(lora)
        self._fisher = fisher
        return n_batches
