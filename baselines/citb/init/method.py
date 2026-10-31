from __future__ import annotations

from typing import Any, Dict

from core.data import Segment

from baselines.citb.common import empty_train_metrics, ensure_default_adapter


class CITBInitMethod:
    """Frozen Instruct (untrained LoRA B≈0). Eval-only on the CL stream; F=0."""

    name = "citb_init"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        active = ensure_default_adapter(lora)
        if hasattr(lora, "freeze_adapter") and active in (lora.list_adapters() if hasattr(lora, "list_adapters") else []):
            lora.freeze_adapter(active)
        return {"active_adapter": active, "eval_only": True}

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
        _ = (segment, model, lr, epochs, batch_size)
        metrics = empty_train_metrics(extra={"active_adapter": lora.get_active_adapter_name()})
        return metrics
