from __future__ import annotations

import random
from typing import Any, Dict, Optional

from core.data import Segment

from baselines.citb.common import (
    cfg_seed,
    empty_train_metrics,
    ensure_default_adapter,
    run_ce_sft,
    segment_train_examples,
    to_pairs,
)


class CITBMultiMethod:
    """Joint LoRA SFT on the full stream at segment 0; later segments eval only."""

    name = "citb_multi"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._stream: Any = None
        self._trained = False
        self._rng = random.Random(cfg_seed(cfg))

    def bind_stream(self, stream: Any) -> None:
        self._stream = stream

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "multi_already_trained": bool(self._trained),
            "is_joint_train_segment": int(getattr(segment, "segment_id", -1)) == 0 and not self._trained,
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
        if self._trained or int(getattr(segment, "segment_id", -1)) != 0:
            return empty_train_metrics(
                extra={
                    "active_adapter": lora.get_active_adapter_name(),
                    "multi_joint_examples": 0,
                }
            )

        examples = segment_train_examples(self._stream) if self._stream is not None else list(segment.train)
        if not examples:
            examples = list(segment.train)
        pairs, targets = to_pairs(examples)
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
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["multi_joint_examples"] = len(pairs)
        self._trained = True
        return metrics
