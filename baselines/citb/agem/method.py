from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch

from core.data import Segment

from baselines.citb.common import (
    assign_grads,
    batch_pairs,
    cfg_seed,
    ensure_default_adapter,
    flatten_grads,
    iter_trainable_lora_params,
    replay_k,
    sample_k,
    summarize_sft_metrics,
    to_pairs,
    zero_lora_grad,
)


class CITBAGEMMethod:
    """A-GEM: project current grads onto memory grads when the angle is negative."""

    name = "citb_agem"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.k = replay_k(cfg, default=50)
        self._rng = random.Random(cfg_seed(cfg))
        self._memory_pairs: List[Tuple[str, str]] = []
        self._memory_targets: List[str] = []
        self._first_task = True

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "agem_k": self.k,
            "agem_memory_size": len(self._memory_pairs),
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
        n_projected = 0
        use_agem = bool(self._memory_pairs) and not self._first_task
        model._last_lr = float(lr)

        for _ in range(max(1, int(epochs))):
            idxs = list(range(len(pairs)))
            if len(idxs) > 1:
                self._rng.shuffle(idxs)
            order_pairs = [pairs[i] for i in idxs]
            order_targets = [targets[i] for i in idxs]
            for b_pairs, b_targets in batch_pairs(order_pairs, order_targets, batch_size):
                g_ref = None
                if use_agem:
                    mem_pairs, mem_targets = self._sample_memory_batch(batch_size)
                    if mem_pairs:
                        zero_lora_grad(lora)
                        mem_loss, _ = model.compute_supervised_loss(mem_pairs, mem_targets)
                        mem_loss.backward()
                        g_ref = flatten_grads(iter_trainable_lora_params(lora)).clone()
                        zero_lora_grad(lora)

                ce_loss, stats = model.compute_supervised_loss(b_pairs, b_targets)
                ce_loss.backward()
                if g_ref is not None:
                    params = iter_trainable_lora_params(lora)
                    g_cur = flatten_grads(params)
                    if g_cur.numel() == g_ref.numel():
                        angle = float((g_cur * g_ref).sum().item())
                        if angle < 0.0:
                            length_rep = float((g_ref * g_ref).sum().item())
                            if length_rep > 0.0:
                                g_proj = g_cur - (angle / length_rep) * g_ref
                                assign_grads(params, g_proj)
                                n_projected += 1

                step_stats = lora.step_adapter()
                batch_accs.append(float(stats.get("train_batch_acc", 0.0)))
                batch_losses.append(float(stats.get("train_loss", 0.0)))
                answer_token_accs.append(float(stats.get("train_answer_token_acc", 0.0)))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                total_tokens += int(stats.get("num_total_tokens", 0))
                supervised_tokens += int(stats.get("num_supervised_tokens", 0))
                batches += 1

        mem_pairs, mem_targets = sample_k(pairs, targets, self.k, self._rng)
        self._memory_pairs.extend(mem_pairs)
        self._memory_targets.extend(mem_targets)
        self._first_task = False
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
                "agem_k": self.k,
                "agem_applied": bool(use_agem),
                "agem_projected_steps": int(n_projected),
                "agem_memory_size": len(self._memory_pairs),
            },
        )
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics

    def _sample_memory_batch(self, batch_size: int) -> Tuple[List[Tuple[str, str]], List[str]]:
        n = len(self._memory_pairs)
        if n <= 0:
            return [], []
        take = min(max(1, int(batch_size)), n)
        idxs = self._rng.sample(range(n), k=take)
        return [self._memory_pairs[i] for i in idxs], [self._memory_targets[i] for i in idxs]
