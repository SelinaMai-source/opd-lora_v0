from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

from core.data import Segment

from baselines.citb.common import (
    cfg_seed,
    ensure_default_adapter,
    replay_k,
    run_ce_sft,
    sample_k,
    to_pairs,
)


class CITBReplayMethod:
    """Per-task memory of K examples; next segment trains current+memory CE."""

    name = "citb_replay"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.k = replay_k(cfg, default=50)
        self._rng = random.Random(cfg_seed(cfg))
        self._memory_pairs: List[Tuple[str, str]] = []
        self._memory_targets: List[str] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "replay_k": self.k,
            "replay_memory_size": len(self._memory_pairs),
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
        mix_pairs = list(current_pairs) + list(self._memory_pairs)
        mix_targets = list(current_targets) + list(self._memory_targets)
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
        mem_pairs, mem_targets = sample_k(current_pairs, current_targets, self.k, self._rng)
        self._memory_pairs.extend(mem_pairs)
        self._memory_targets.extend(mem_targets)
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["replay_k"] = self.k
        metrics["replay_added"] = len(mem_pairs)
        metrics["replay_memory_size"] = len(self._memory_pairs)
        metrics["replay_mix_examples"] = len(mix_pairs)
        return metrics
