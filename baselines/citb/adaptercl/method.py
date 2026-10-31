from __future__ import annotations

import random
from typing import Any, Dict, List

import torch

from core.data import Segment

from baselines.citb.common import (
    cfg_seed,
    empty_cuda_cache,
    run_ce_sft,
    to_pairs,
)


class CITBAdapterCLMethod:
    """One LoRA per segment; eval switches by oracle source_segment_id."""

    name = "citb_adaptercl"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        citb = cfg.get("citb", {}) if isinstance(cfg.get("citb", {}), dict) else {}
        self.adapter_prefix = str(citb.get("adapter_prefix", "citb_seg"))
        self.cpu_offload_inactive = bool(citb.get("cpu_offload_inactive_adapters", True))
        self.optimizer_active_only = bool(citb.get("optimizer_active_only", True))
        self._rng = random.Random(cfg_seed(cfg))
        self._seg_to_adapter: Dict[int, str] = {}
        self._lora: Any = None
        self._gpu_resident: str = ""
        self._eval_hooked = False

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        self._lora = lora
        self._install_eval_hook(model)
        adapter_name = self._adapter_name(int(segment.segment_id))
        if adapter_name not in lora.list_adapters():
            lora.create_adapter(adapter_name)
        self._seg_to_adapter[int(segment.segment_id)] = adapter_name
        self._freeze_inactive(lora, adapter_name)
        lora.set_active_adapter(adapter_name)
        if hasattr(lora, "unfreeze_adapter"):
            lora.unfreeze_adapter(adapter_name)
        self._prepare_active(lora, adapter_name)
        return {
            "active_adapter": lora.get_active_adapter_name(),
            "num_adapters": len(self._seg_to_adapter),
            "adaptercl_oracle": True,
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
        adapter_name = self._seg_to_adapter.get(int(segment.segment_id), lora.get_active_adapter_name())
        self._prepare_active(lora, adapter_name)
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
        if self.cpu_offload_inactive:
            _move_adapter_params(lora, adapter_name, torch.device("cpu"))
            self._gpu_resident = ""
        empty_cuda_cache()
        metrics["active_adapter"] = adapter_name
        metrics["num_adapters"] = len(self._seg_to_adapter)
        return metrics

    def _adapter_name(self, segment_id: int) -> str:
        return f"{self.adapter_prefix}_{int(segment_id):03d}"

    def _install_eval_hook(self, model: Any) -> None:
        if self._eval_hooked:
            return

        def _on_eval_segment(segment: Segment) -> None:
            lora = self._lora
            if lora is None:
                return
            seg_id = int(getattr(segment, "segment_id", -1))
            name = self._seg_to_adapter.get(seg_id)
            if not name or name not in lora.list_adapters():
                return
            self._safe_set_adapter(lora, name)

        model.on_eval_segment = _on_eval_segment
        self._eval_hooked = True

    def _freeze_inactive(self, lora: Any, active: str) -> None:
        if not hasattr(lora, "freeze_adapter"):
            return
        for name in lora.list_adapters():
            if name != active:
                lora.freeze_adapter(name)

    def _prepare_active(self, lora: Any, adapter_name: str) -> None:
        if self.cpu_offload_inactive:
            for name in lora.list_adapters():
                if name != adapter_name:
                    _move_adapter_params(lora, name, torch.device("cpu"))
        self._ensure_adapter_device(lora, adapter_name)
        lora.set_active_adapter(adapter_name)
        if hasattr(lora, "unfreeze_adapter"):
            lora.unfreeze_adapter(adapter_name)
        if self.optimizer_active_only:
            self._rebuild_active_optimizer(lora, adapter_name)
        self._gpu_resident = adapter_name
        empty_cuda_cache()

    def _safe_set_adapter(self, lora: Any, name: str) -> None:
        adapters = set(lora.list_adapters()) if hasattr(lora, "list_adapters") else set()
        if name not in adapters:
            return
        if self.cpu_offload_inactive:
            prev = self._gpu_resident
            if prev and prev != name and prev in adapters:
                _move_adapter_params(lora, prev, torch.device("cpu"))
        self._ensure_adapter_device(lora, name)
        lora.set_active_adapter(name)

    def _ensure_adapter_device(self, lora: Any, adapter_name: str) -> None:
        device = _lora_compute_device(lora)
        _move_adapter_params(lora, adapter_name, device)
        self._gpu_resident = adapter_name

    def _rebuild_active_optimizer(self, lora: Any, adapter_name: str) -> None:
        if not hasattr(lora, "_optimizer"):
            return
        params = [p for _, p in _iter_adapter_params(lora, adapter_name) if p.requires_grad]
        if not params:
            return
        lora._optimizer = torch.optim.AdamW(params, lr=1e-4)


def _iter_adapter_params(lora: Any, adapter_name: str):
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is None:
        return
    needle_a = f"lora_A.{adapter_name}."
    needle_b = f"lora_B.{adapter_name}."
    for name, param in peft_model.named_parameters():
        if needle_a in name or needle_b in name:
            yield name, param


def _lora_compute_device(lora: Any):
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is None:
        return None
    for param in peft_model.parameters():
        if param.device.type == "cuda":
            return param.device
    try:
        return next(peft_model.parameters()).device
    except StopIteration:
        return None


def _move_adapter_params(lora: Any, adapter_name: str, device: Any) -> None:
    if device is None:
        return
    with torch.no_grad():
        for _, param in _iter_adapter_params(lora, adapter_name):
            if param.device != device:
                param.data = param.data.to(device, non_blocking=False)
