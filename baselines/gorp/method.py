from __future__ import annotations

"""GORP (ACL 2025): project LoRA grads onto a unified low-rank SVD subspace.

Logic follows official `src/Proj_torch/proj_projector.py` (ProjProjector std).
Applied to each trainable LoRA matrix after CE backward, before step_adapter.
"""

import random
from typing import Any, Dict, List, Optional

import torch

from core.data import Segment

from baselines.citb.common import (
    batch_pairs,
    cfg_seed,
    ensure_default_adapter,
    summarize_sft_metrics,
    to_pairs,
    zero_lora_grad,
)
from baselines.lit_common import method_cfg


class _ProjProjector:
    def __init__(self, rank: int, update_proj_gap: int = 50, scale: float = 1.0):
        self.rank = max(1, int(rank))
        self.update_proj_gap = max(1, int(update_proj_gap))
        self.scale = float(scale)
        self.ortho_matrix: Optional[torch.Tensor] = None
        self._side = "right"

    def project(self, full_rank_grad: torch.Tensor, step: int) -> torch.Tensor:
        g = full_rank_grad
        if g.ndim != 2 or min(g.shape) == 0:
            return g
        if g.shape[0] >= g.shape[1]:
            self._side = "right"
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                self.ortho_matrix = self._ortho(g, "right")
            low = torch.matmul(g, self.ortho_matrix.t())
        else:
            self._side = "left"
            if self.ortho_matrix is None or step % self.update_proj_gap == 0:
                self.ortho_matrix = self._ortho(g, "left")
            low = torch.matmul(self.ortho_matrix.t(), g)
        return low

    def project_back(self, low_rank_grad: torch.Tensor) -> torch.Tensor:
        if self.ortho_matrix is None:
            return low_rank_grad
        if self._side == "right":
            full = torch.matmul(low_rank_grad, self.ortho_matrix)
        else:
            full = torch.matmul(self.ortho_matrix, low_rank_grad)
        return full * self.scale

    def _ortho(self, weights: torch.Tensor, side: str) -> torch.Tensor:
        matrix = weights.detach().float()
        rank = min(self.rank, min(matrix.shape))
        _u, _s, vh = torch.linalg.svd(matrix, full_matrices=False)
        if side == "right":
            b = vh[:rank, :]
        else:
            b = _u[:, :rank]
        return b.to(device=weights.device, dtype=weights.dtype)


class GORPMethod:
    name = "gorp"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "gorp")
        self.proj_rank = int(block.get("proj_rank", 8))
        self.update_proj_gap = int(block.get("update_proj_gap", 50))
        self._rng = random.Random(cfg_seed(cfg))
        self._projectors: Dict[str, _ProjProjector] = {}
        self._step = 0

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "gorp_proj_rank": self.proj_rank,
            "gorp_update_proj_gap": self.update_proj_gap,
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
                ce_loss.backward()
                n_projected += self._project_grads(lora)
                step_stats = lora.step_adapter()
                self._step += 1
                batch_accs.append(float(stats.get("train_batch_acc", 0.0)))
                batch_losses.append(float(stats.get("train_loss", 0.0)))
                answer_token_accs.append(float(stats.get("train_answer_token_acc", 0.0)))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                total_tokens += int(stats.get("num_total_tokens", 0))
                supervised_tokens += int(stats.get("num_supervised_tokens", 0))
                batches += 1

        return summarize_sft_metrics(
            batches=batches,
            batch_accs=batch_accs,
            batch_losses=batch_losses,
            answer_token_accs=answer_token_accs,
            grad_norms=grad_norms,
            lr_values=lr_values,
            total_tokens=total_tokens,
            supervised_tokens=supervised_tokens,
            extra={
                "gorp_projected_steps": n_projected,
                "gorp_proj_rank": self.proj_rank,
                "active_adapter": lora.get_active_adapter_name(),
            },
        )

    def _project_grads(self, lora: Any) -> int:
        peft_model = getattr(lora, "peft_model", None)
        if peft_model is None:
            return 0
        n = 0
        for name, param in peft_model.named_parameters():
            if "lora_" not in name or not param.requires_grad or param.grad is None:
                continue
            g = param.grad
            if g.ndim != 2:
                continue
            proj = self._projectors.get(name)
            if proj is None:
                proj = _ProjProjector(self.proj_rank, self.update_proj_gap)
                self._projectors[name] = proj
            low = proj.project(g, self._step)
            full = proj.project_back(low)
            param.grad.copy_(full.to(dtype=param.grad.dtype))
            n += 1
        return n
