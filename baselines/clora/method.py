from __future__ import annotations

"""CLoRA (ACL 2025): PEFT LoRA + frozen orthogonal subspace regularizer.

Adapted from official `baselines/clora/official/clora.py` (CLoraWrapper).
Does not run the official trainer. k is the subspace width on base Linear
in/out dims; lambda_ weights ||A V||_F^2 / ||B^T U||_F^2 added to CE.
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

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


class CLoRAMethod:
    name = "clora"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "clora")
        self.k = int(block.get("k", 128))
        self.lambda_reg = float(block.get("lambda_", block.get("lambda", 1.0)))
        self._rng = __import__("random").Random(cfg_seed(cfg))
        self._u: Dict[str, torch.Tensor] = {}
        self._v: Dict[str, torch.Tensor] = {}
        self._prepared = False

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        self._maybe_init_bases(lora)
        return {
            "active_adapter": active,
            "clora_k": self.k,
            "clora_lambda": self.lambda_reg,
            "clora_num_bases": len(self._u),
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
        self._maybe_init_bases(lora)
        pairs, targets = to_pairs(segment.train)
        batch_accs: List[float] = []
        batch_losses: List[float] = []
        answer_token_accs: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        reg_vals: List[float] = []
        total_tokens = 0
        supervised_tokens = 0
        batches = 0
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
                reg = self._reg_loss(lora)
                loss = ce_loss + self.lambda_reg * reg
                loss.backward()
                step_stats = lora.step_adapter()
                batch_accs.append(float(stats.get("train_batch_acc", 0.0)))
                batch_losses.append(float(stats.get("train_loss", 0.0)))
                answer_token_accs.append(float(stats.get("train_answer_token_acc", 0.0)))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                reg_vals.append(float(reg.detach().item()) if torch.is_tensor(reg) else float(reg))
                total_tokens += int(stats.get("num_total_tokens", 0))
                supervised_tokens += int(stats.get("num_supervised_tokens", 0))
                batches += 1

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
                "clora_reg_mean": sum(reg_vals) / max(1, len(reg_vals)),
                "clora_k": self.k,
                "clora_lambda": self.lambda_reg,
                "active_adapter": lora.get_active_adapter_name(),
            },
        )
        return metrics

    def _maybe_init_bases(self, lora: Any) -> None:
        if self._prepared:
            return
        peft_model = getattr(lora, "peft_model", None)
        if peft_model is None:
            return
        # Match official CLoraWrapper: one frozen orthogonal U/V per target Linear.
        for name, module in peft_model.named_modules():
            if not hasattr(module, "base_layer"):
                continue
            weight = getattr(getattr(module, "base_layer", None), "weight", None)
            if weight is None or weight.ndim != 2:
                continue
            key = name.replace(".", "-")
            out_f, in_f = int(weight.size(0)), int(weight.size(1))
            u = torch.empty(out_f, self.k, device=weight.device, dtype=torch.float32)
            v = torch.empty(in_f, self.k, device=weight.device, dtype=torch.float32)
            nn.init.orthogonal_(u)
            nn.init.orthogonal_(v)
            self._u[key] = u.detach()
            self._v[key] = v.detach()
        self._prepared = True

    def _lookup_uv(self, peft_key: str) -> Tuple[str, torch.Tensor] | Tuple[None, None]:
        key = peft_key.replace(".", "-")
        table = self._u if "lora_B" in key else self._v
        for k, mat in table.items():
            if k in key:
                return k, mat
        return None, None

    def _reg_loss(self, lora: Any) -> torch.Tensor:
        peft_model = getattr(lora, "peft_model", None)
        device = None
        total = None
        if peft_model is None:
            return torch.zeros((), dtype=torch.float32)
        for n, p in peft_model.named_parameters():
            if "lora_" not in n or not p.requires_grad:
                continue
            _, reg_p = self._lookup_uv(n)
            if reg_p is None:
                continue
            mat = p.float()
            basis = reg_p.to(device=mat.device, dtype=mat.dtype)
            if "lora_A" in n:
                # p: r x in ; V: in x k  -> r x k
                reg_matrix = torch.matmul(mat, basis)
            elif "lora_B" in n:
                # p: out x r ; U: out x k -> r x k
                reg_matrix = torch.matmul(mat.T, basis)
            else:
                continue
            term = torch.norm(reg_matrix, p="fro") ** 2 / 2
            total = term if total is None else total + term
            device = mat.device
        if total is None:
            return torch.zeros((), device=device or "cpu", dtype=torch.float32)
        return total
