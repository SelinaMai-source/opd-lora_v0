from __future__ import annotations

"""TreeLoRA (ICML 2025): per-segment LoRA + gradient-similarity tree regularizer.

Does not rerun official DeepSpeed. Layer-wise (q_proj vs v_proj) mean LoRA-A
grads build a KD-style split; selected previous task grads get a negative-dot
regularizer as in `official/utils/kd_lora_tree.py` (`tree_lora_loss`).
Eval uses oracle segment adapters.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from core.data import Segment

from baselines.citb.common import (
    batch_pairs,
    summarize_sft_metrics,
    to_pairs,
    zero_lora_grad,
)
from baselines.lit_common import cfg_seed, install_oracle_eval_hook, method_cfg, spawn_segment_adapter


class TreeLoRAMethod:
    name = "treelora"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "treelora")
        self.adapter_prefix = str(block.get("adapter_prefix", "tree_s"))
        self.reg = float(block.get("reg", 0.1))
        self._rng = random.Random(cfg_seed(cfg))
        self._seg_to_adapter: Dict[int, str] = {}
        self._layer_grads: List[Dict[str, torch.Tensor]] = []
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
            "treelora_reg": self.reg,
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
        pairs, targets = to_pairs(segment.train)
        batch_accs: List[float] = []
        batch_losses: List[float] = []
        answer_token_accs: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        total_tokens = 0
        supervised_tokens = 0
        batches = 0
        acc_layers: Dict[str, torch.Tensor] = {}
        n_acc = 0
        task_id = len(self._layer_grads)
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
                layer_vec = self._layer_param_stack(lora, name)
                if task_id > 0 and layer_vec is not None:
                    prev_id = self._tree_select(layer_vec)
                    prev = self._layer_grads[prev_id]
                    terms = []
                    for key, cur in layer_vec.items():
                        old = prev.get(key)
                        if old is None:
                            continue
                        old_t = old.to(device=cur.device, dtype=cur.dtype)
                        terms.append(-(cur.reshape(-1) * old_t.reshape(-1)).sum())
                    if terms:
                        reg = torch.stack(terms).mean()
                        ce_loss = ce_loss - self.reg * reg
                ce_loss.backward()
                snap = self._layer_grad_stack(lora, name)
                if snap:
                    for k, v in snap.items():
                        acc_layers[k] = v if k not in acc_layers else acc_layers[k] + v
                    n_acc += 1
                step_stats = lora.step_adapter()
                batch_accs.append(float(stats.get("train_batch_acc", 0.0)))
                batch_losses.append(float(stats.get("train_loss", 0.0)))
                answer_token_accs.append(float(stats.get("train_answer_token_acc", 0.0)))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                total_tokens += int(stats.get("num_total_tokens", 0))
                supervised_tokens += int(stats.get("num_supervised_tokens", 0))
                batches += 1

        if n_acc > 0:
            self._layer_grads.append({k: v / n_acc for k, v in acc_layers.items()})
        else:
            self._layer_grads.append({})
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
                "treelora_task_id": task_id,
                "num_adapters": len(self._seg_to_adapter),
                "active_adapter": name,
            },
        )

    def _keys(self, name: str, param_name: str) -> Optional[str]:
        if f"lora_A.{name}." not in param_name:
            return None
        if "q_proj" in param_name:
            return "q_proj"
        if "v_proj" in param_name:
            return "v_proj"
        return "other"

    def _layer_param_stack(self, lora: Any, adapter_name: str) -> Optional[Dict[str, torch.Tensor]]:
        peft_model = getattr(lora, "peft_model", None)
        if peft_model is None:
            return None
        buckets: Dict[str, List[torch.Tensor]] = {}
        for n, p in peft_model.named_parameters():
            key = self._keys(adapter_name, n)
            if key is None:
                continue
            buckets.setdefault(key, []).append(p.reshape(-1))
        if not buckets:
            return None
        return {k: torch.cat(vs) for k, vs in buckets.items()}

    def _layer_grad_stack(self, lora: Any, adapter_name: str) -> Dict[str, torch.Tensor]:
        peft_model = getattr(lora, "peft_model", None)
        out: Dict[str, torch.Tensor] = {}
        if peft_model is None:
            return out
        buckets: Dict[str, List[torch.Tensor]] = {}
        for n, p in peft_model.named_parameters():
            key = self._keys(adapter_name, n)
            if key is None or p.grad is None:
                continue
            buckets.setdefault(key, []).append(p.grad.detach().float().reshape(-1).cpu())
        for k, vs in buckets.items():
            out[k] = torch.cat(vs)
        return out

    def _tree_select(self, layer_vec: Dict[str, torch.Tensor]) -> int:
        if not self._layer_grads:
            return 0
        # Median split on concatenated layer grads (kd_lora_tree one-step analogue).
        scores = []
        cur = torch.cat([v.detach().float().cpu() for v in layer_vec.values()]) if layer_vec else None
        for i, past in enumerate(self._layer_grads):
            if not past:
                scores.append((0.0, i))
                continue
            old = torch.cat([past[k] for k in past])
            if cur is None:
                scores.append((0.0, i))
                continue
            n = min(cur.numel(), old.numel())
            sim = float(torch.dot(cur[:n], old[:n]).item()) if n else 0.0
            scores.append((sim, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return int(scores[0][1])
