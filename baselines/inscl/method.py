from __future__ import annotations

"""InsCL (NAACL 2024): Wasserstein replay allocation + InsInfo sampling.

Official sampling lives in `official/replay/InsCL_sampling.py`. GPT-4 tags are
replaced by a length/diversity InsInfo proxy. Memory K=10 per past task budget
is reallocated by 1-Wasserstein on hashed instruction embeddings.
"""

import random
from typing import Any, Dict, List, Tuple

from core.data import Segment

from baselines.citb.common import run_ce_sft, to_pairs
from baselines.lit_common import (
    allocate_by_wasserstein,
    cfg_seed,
    ensure_default_adapter,
    hashed_bow,
    method_cfg,
    replay_k,
    sample_by_insinfo,
)


class InsCLMethod:
    name = "inscl"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.k = replay_k(cfg, "inscl", default=10)
        self._rng = random.Random(cfg_seed(cfg))
        self._past: List[Dict[str, Any]] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "inscl_k": self.k,
            "inscl_n_past": len(self._past),
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
        current_pairs, current_targets = to_pairs(segment.train)
        mix_pairs = list(current_pairs)
        mix_targets = list(current_targets)
        replay_added = 0
        alloc: List[int] = []
        if self._past:
            cur_emb = hashed_bow([p[0] for p in current_pairs] or [str(segment.segment_name)], dim=128)
            past_embs = [item["emb"] for item in self._past]
            total = self.k * len(self._past)
            alloc = allocate_by_wasserstein(cur_emb, past_embs, total)
            for n_take, item in zip(alloc, self._past):
                if n_take <= 0:
                    continue
                rp, rt = sample_by_insinfo(item["pairs"], item["targets"], n_take, self._rng)
                mix_pairs.extend(rp)
                mix_targets.extend(rt)
                replay_added += len(rp)

        metrics = run_ce_sft(
            model=model,
            lora=lora,
            pairs=mix_pairs,
            targets=mix_targets,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            rng=self._rng,
            shuffle=True,
        )
        self._past.append(
            {
                "pairs": list(current_pairs),
                "targets": list(current_targets),
                "emb": hashed_bow([p[0] for p in current_pairs] or [str(segment.segment_name)], dim=128),
            }
        )
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["inscl_k"] = self.k
        metrics["inscl_replay_added"] = replay_added
        metrics["inscl_alloc"] = alloc
        metrics["replay_mix_examples"] = len(mix_pairs)
        return metrics
