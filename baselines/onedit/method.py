from __future__ import annotations

"""OnEDIT (AAAI 2026): fixed implicit-task adapters, no task-id.

No public official repo. Paper-faithful: a fixed pool of implicit adapters;
each segment's instruction embedding is assigned to the nearest slot (or an
empty slot until capacity). Train only that adapter. Eval routes by centroid
similarity. Data still arrives at segment boundaries in this setting.
"""

import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from core.data import Segment

from baselines.citb.common import run_ce_sft, to_pairs
from baselines.lit_common import (
    cfg_seed,
    hashed_bow,
    mean_embedding,
    method_cfg,
    prepare_train_adapter,
    spawn_segment_adapter,
    wrap_generate_router,
)


class OnEDITMethod:
    name = "onedit"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "onedit")
        self.n_slots = max(1, int(block.get("n_slots", 4)))
        self.sim_threshold = float(block.get("sim_threshold", 0.35))
        self.adapter_prefix = str(block.get("adapter_prefix", "onedit_i"))
        self._rng = random.Random(cfg_seed(cfg))
        self._slots: List[str] = []
        self._centroids: Dict[str, torch.Tensor] = {}
        self._counts: Dict[str, int] = {}
        self._lora: Any = None
        self._eval_wrapped = False
        self._active: Optional[str] = None

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        self._lora = lora
        self._install_router(model)
        pairs, _ = to_pairs(segment.train)
        q = mean_embedding([p[0] for p in pairs] or [str(segment.segment_name)])
        name, reused = self._assign_slot(lora, q)
        self._active = name
        lora.set_active_adapter(name)
        if hasattr(lora, "freeze_adapter"):
            for other in lora.list_adapters():
                if other != name:
                    lora.freeze_adapter(other)
        if hasattr(lora, "unfreeze_adapter"):
            lora.unfreeze_adapter(name)
        return {
            "active_adapter": name,
            "onedit_reused": reused,
            "onedit_n_slots": len(self._slots),
            "onedit_capacity": self.n_slots,
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
        if self._active:
            prepare_train_adapter(lora, self._active, model)
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
        name = lora.get_active_adapter_name()
        new_c = mean_embedding([p[0] for p in pairs] or [str(segment.segment_name)])
        old = self._centroids.get(name)
        cnt = self._counts.get(name, 0)
        if old is None:
            self._centroids[name] = new_c
        else:
            mixed = (old * cnt + new_c) / float(cnt + 1)
            n = float(mixed.norm().item())
            self._centroids[name] = mixed / n if n > 0 else mixed
        self._counts[name] = cnt + 1
        metrics["active_adapter"] = name
        metrics["onedit_n_slots"] = len(self._slots)
        return metrics

    def _assign_slot(self, lora: Any, q: torch.Tensor) -> tuple[str, bool]:
        qn = F.normalize(q, dim=0, eps=1e-8)
        best_name = None
        best_sim = -1e9
        for name in self._slots:
            c = self._centroids.get(name)
            if c is None:
                continue
            sim = float(torch.dot(qn, F.normalize(c, dim=0, eps=1e-8)).item())
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_name is not None and best_sim >= self.sim_threshold:
            return best_name, True
        if len(self._slots) < self.n_slots:
            name = f"{self.adapter_prefix}{len(self._slots)}"
            spawn_segment_adapter(lora, name, freeze_others=True)
            self._slots.append(name)
            self._centroids[name] = qn
            self._counts[name] = 0
            return name, False
        return best_name or self._slots[0], True

    def _install_router(self, model: Any) -> None:
        def _route(prompts: List[str]) -> None:
            lora = self._lora
            if lora is None or not self._slots:
                return
            q = hashed_bow(prompts or [""], dim=256).mean(dim=0)
            q = F.normalize(q, dim=0, eps=1e-8)
            best = self._slots[0]
            best_sim = -1e9
            for name in self._slots:
                c = self._centroids.get(name)
                if c is None:
                    continue
                sim = float(torch.dot(q, F.normalize(c, dim=0, eps=1e-8)).item())
                if sim > best_sim:
                    best_sim = sim
                    best = name
            lora.set_active_adapter(best)

        wrap_generate_router(self, model, _route)
