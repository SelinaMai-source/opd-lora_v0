from __future__ import annotations

"""SABER (ICML 2026): replay-free backward refine along non-interfering LoRA directions.

Official decoder uses prompt SVD + proj_cos. Here each segment has a LoRA;
related old adapters (projection>=0.1 and compatibility>0) are updated with
gradients projected off their SVD protected basis (`_safe_backward_step`).
Eval routes by source_segment_id (oracle, same as CITB AdapterCL).
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.data import Segment

from baselines.citb.common import (
    batch_pairs,
    flatten_grads,
    iter_trainable_lora_params,
    run_ce_sft,
    summarize_sft_metrics,
    to_pairs,
    zero_lora_grad,
)
from baselines.lit_common import (
    cfg_seed,
    install_oracle_eval_hook,
    lora_named_params,
    method_cfg,
    spawn_segment_adapter,
)


class SABERMethod:
    name = "saber"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "saber")
        self.adapter_prefix = str(block.get("adapter_prefix", "saber_s"))
        self.proj_thresh = float(block.get("proj_thresh", 0.1))
        self.backward_lr = float(block.get("backward_lr", 1e-3))
        self.refine_steps = int(block.get("refine_steps", 8))
        self._rng = random.Random(cfg_seed(cfg))
        self._seg_to_adapter: Dict[int, str] = {}
        self._mean_grads: Dict[str, torch.Tensor] = {}
        self._basis: Dict[str, torch.Tensor] = {}
        self._lora: Any = None

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        self._lora = lora
        install_oracle_eval_hook(
            self,
            model,
            lambda seg: self._seg_to_adapter.get(int(getattr(seg, "segment_id", -1))),
        )
        name = f"{self.adapter_prefix}{int(segment.segment_id):03d}"
        spawn_segment_adapter(lora, name, freeze_others=True)
        self._seg_to_adapter[int(segment.segment_id)] = name
        return {
            "active_adapter": lora.get_active_adapter_name(),
            "num_adapters": len(self._seg_to_adapter),
            "saber_oracle": True,
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
        self._lora = lora
        name = self._seg_to_adapter[int(segment.segment_id)]
        lora.set_active_adapter(name)
        if hasattr(lora, "unfreeze_adapter"):
            lora.unfreeze_adapter(name)
        pairs, targets = to_pairs(segment.train)
        metrics = run_ce_sft(
            model=model,
            lora=lora,
            pairs=pairs,
            targets=targets,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            rng=self._rng,
            shuffle=True,
        )
        g_mean = self._capture_mean_grad(model, lora, pairs, targets, batch_size)
        if g_mean is not None:
            self._mean_grads[name] = g_mean
        self._basis[name] = self._svd_basis(lora, name)
        n_ref = self._backward_refine(model, lora, name, pairs, targets, batch_size)
        metrics["active_adapter"] = name
        metrics["saber_refined"] = n_ref
        metrics["num_adapters"] = len(self._seg_to_adapter)
        return metrics

    def _capture_mean_grad(
        self,
        model: Any,
        lora: Any,
        pairs: List[Tuple[str, str]],
        targets: List[str],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        acc = None
        n = 0
        model._last_lr = 0.0
        for b_pairs, b_targets in batch_pairs(pairs, targets, batch_size):
            zero_lora_grad(lora)
            loss, _ = model.compute_supervised_loss(b_pairs, b_targets)
            loss.backward()
            g = flatten_grads(iter_trainable_lora_params(lora)).detach().float().cpu()
            acc = g if acc is None else acc + g
            n += 1
            zero_lora_grad(lora)
            if n >= 4:
                break
        if acc is None or n <= 0:
            return None
        v = acc / n
        nrm = float(v.norm().item())
        if nrm > 0:
            v = v / nrm
        return v

    def _svd_basis(self, lora: Any, adapter_name: str, rank: int = 8) -> torch.Tensor:
        vecs = []
        for _, p in lora_named_params(lora, adapter_name):
            mat = p.detach().float().cpu()
            if mat.ndim != 2 or min(mat.shape) == 0:
                continue
            try:
                _u, _s, vh = torch.linalg.svd(mat, full_matrices=False)
            except RuntimeError:
                continue
            k = min(int(rank), int(_s.numel()))
            if k <= 0:
                continue
            vecs.append(vh[:k, :].reshape(k, -1))
        if not vecs:
            return torch.zeros(1, 1)
        cat = torch.cat(vecs, dim=1)
        q, _ = torch.linalg.qr(cat.T, mode="reduced")
        return q[:, : min(16, q.shape[1])].contiguous()

    def _select_related(self, current: str) -> List[str]:
        g_cur = self._mean_grads.get(current)
        if g_cur is None:
            return []
        selected = []
        for name, g_old in self._mean_grads.items():
            if name == current:
                continue
            n = min(g_cur.numel(), g_old.numel())
            if n <= 0:
                continue
            a = g_cur[:n]
            b = g_old[:n]
            compat = float((a * b).sum().item())
            proj = abs(compat)
            if proj >= self.proj_thresh and compat > 0.0:
                selected.append(name)
        return selected

    def _backward_refine(
        self,
        model: Any,
        lora: Any,
        current: str,
        pairs: List[Tuple[str, str]],
        targets: List[str],
        batch_size: int,
    ) -> int:
        related = self._select_related(current)
        if not related or not pairs:
            return 0
        n_ok = 0
        for old in related:
            if hasattr(lora, "unfreeze_adapter"):
                lora.unfreeze_adapter(old)
            lora.set_active_adapter(old)
            basis = self._basis.get(old)
            for step in range(max(1, self.refine_steps)):
                i0 = self._rng.randrange(len(pairs))
                i1 = min(len(pairs), i0 + max(1, batch_size))
                zero_lora_grad(lora)
                loss, _ = model.compute_supervised_loss(pairs[i0:i1], targets[i0:i1])
                loss.backward()
                self._project_off_basis(lora, old, basis)
                with torch.no_grad():
                    for _, p in lora_named_params(lora, old):
                        if p.grad is None:
                            continue
                        p.add_(p.grad, alpha=-self.backward_lr)
                zero_lora_grad(lora)
            if hasattr(lora, "freeze_adapter"):
                lora.freeze_adapter(old)
            n_ok += 1
        lora.set_active_adapter(current)
        if hasattr(lora, "unfreeze_adapter"):
            lora.unfreeze_adapter(current)
        return n_ok

    def _project_off_basis(self, lora: Any, adapter_name: str, basis: Optional[torch.Tensor]) -> None:
        if basis is None or basis.numel() <= 1:
            return
        params = [(n, p) for n, p in lora_named_params(lora, adapter_name) if p.grad is not None]
        if not params:
            return
        flat = torch.cat([p.grad.detach().float().reshape(-1).cpu() for _, p in params])
        q = basis
        m = min(flat.numel(), q.shape[0])
        if m <= 0:
            return
        g = flat[:m]
        q_m = q[:m].float()
        g = g - q_m @ (q_m.T @ g)
        offset = 0
        with torch.no_grad():
            for _, p in params:
                n = int(p.numel())
                chunk = g[offset : offset + n] if offset < m else torch.zeros(n)
                if chunk.numel() < n:
                    pad = torch.zeros(n)
                    pad[: chunk.numel()] = chunk
                    chunk = pad
                p.grad.copy_(chunk.reshape_as(p.grad).to(device=p.grad.device, dtype=p.grad.dtype))
                offset += n
