from __future__ import annotations

import random
from typing import Any, Dict

from core.data import Segment

from baselines.citb.common import (
    cfg_seed,
    ensure_default_adapter,
    run_ce_sft,
    set_include_instruction,
    to_pairs,
)


class CITBFTNoInstrMethod:
    """Sequential LoRA with task definition stripped from train and eval."""

    name = "citb_ft_no_instr"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self._rng = random.Random(cfg_seed(cfg))

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = segment
        set_include_instruction(model, include=False)
        active = ensure_default_adapter(lora)
        return {"active_adapter": active, "include_instruction": False}

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
        set_include_instruction(model, include=False)
        pairs, targets = to_pairs(segment.train, strip_instruction=True)
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
        metrics["include_instruction"] = False
        return metrics
