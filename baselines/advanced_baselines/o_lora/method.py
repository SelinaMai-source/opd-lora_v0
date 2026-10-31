from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.data import Example, Segment


class OLoraMethod:
    """
    Scaffold for O-LoRA.

    This is smoke-test-ready wiring for the unified baseline loop. It creates a
    fresh adapter per segment, freezes older adapters when supported, and exposes
    an executable orthogonal regularization/projection path against old adapters.
    """

    name = "o_lora"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        method_cfg = _method_cfg(cfg, "o_lora")
        self.adapter_prefix = str(method_cfg.get("adapter_prefix", "olora_s"))
        self.freeze_previous_adapters = bool(method_cfg.get("freeze_previous_adapters", True))
        self.orthogonal_penalty_weight = float(method_cfg.get("orthogonal_penalty_weight", 0.0))
        self.projection_strength = float(method_cfg.get("orthogonal_projection_strength", 0.0))
        self._adapter_names: List[str] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        adapter_name = f"{self.adapter_prefix}{segment.segment_id}"
        if adapter_name not in lora.list_adapters():
            lora.create_adapter(adapter_name)

        if self.freeze_previous_adapters:
            for old_adapter in self._adapter_names:
                if old_adapter in lora.list_adapters() and hasattr(lora, "freeze_adapter"):
                    lora.freeze_adapter(old_adapter)

        lora.set_active_adapter(adapter_name)
        if adapter_name not in self._adapter_names:
            self._adapter_names.append(adapter_name)

        return {
            "active_adapter": lora.get_active_adapter_name(),
            "num_adapters": len(self._adapter_names),
            "reference_adapters": len(self._reference_adapters(adapter_name)),
            "orthogonal_penalty_weight": self.orthogonal_penalty_weight,
            "orthogonal_projection_strength": self.projection_strength,
            "scaffold_status": "orthogonal_regularization_projection_scaffold_ready",
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
        pairs, targets = _to_pairs(segment.train)
        metrics: Dict[str, Any] = {"batches": 0, "mean_batch_acc": 0.0}
        batch_accs: List[float] = []
        batch_losses: List[float] = []
        orthogonal_penalties: List[float] = []
        projection_calls = 0
        projected_calls = 0

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                penalty = self._apply_orthogonal_regularization(lora)
                if penalty is not None:
                    orthogonal_penalties.append(float(penalty))
                projection_stats = self._project_gradients(lora)
                projection_calls += int(bool(projection_stats.get("hook_available", False)))
                projected_calls += int(bool(projection_stats.get("projected", False)))
                step_stats = lora.step_adapter()
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                batch_losses.append(float(out.get("train_loss", 0.0)))
                metrics["batches"] += 1

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["num_adapters"] = len(self._adapter_names)
        metrics["orthogonal_penalty_weight"] = self.orthogonal_penalty_weight
        metrics["orthogonal_projection_strength"] = self.projection_strength
        metrics["orthogonal_penalty_mean"] = sum(orthogonal_penalties) / max(1, len(orthogonal_penalties))
        metrics["orthogonal_projection_hook_calls"] = int(projection_calls)
        metrics["orthogonal_projection_applied_calls"] = int(projected_calls)
        metrics["reference_adapters"] = len(self._reference_adapters(lora.get_active_adapter_name()))
        metrics["grad_norm"] = float(step_stats.get("grad_norm", 0.0)) if "step_stats" in locals() else 0.0
        metrics["scaffold_status"] = "smoke_test_ready_with_orthogonal_hook"
        return metrics

    def _reference_adapters(self, active_adapter: str) -> List[str]:
        return [name for name in self._adapter_names if name != active_adapter]

    def _apply_orthogonal_regularization(self, lora: Any) -> float | None:
        if self.orthogonal_penalty_weight <= 0 or not hasattr(lora, "get_adapter_vector"):
            return None

        try:
            import torch
        except Exception:
            return None

        active = lora.get_active_adapter_name()
        refs = self._reference_adapters(active)
        if not refs:
            return None

        active_vec = lora.get_adapter_vector(active, detach=False).float()
        if active_vec.numel() == 0:
            return None

        penalties = []
        for ref in refs:
            ref_vec = lora.get_adapter_vector(ref, detach=True).to(active_vec.device).float()
            width = min(int(active_vec.numel()), int(ref_vec.numel()))
            if width == 0:
                continue
            denom = active_vec[:width].norm() * ref_vec[:width].norm()
            if float(denom.detach().item()) <= 1e-12:
                continue
            penalties.append(torch.abs(torch.dot(active_vec[:width], ref_vec[:width]) / denom.clamp_min(1e-12)))
        if not penalties:
            return None

        penalty = torch.stack(penalties).mean()
        if penalty.requires_grad:
            (penalty * self.orthogonal_penalty_weight).backward()
        return float(penalty.detach().item())

    def _project_gradients(self, lora: Any) -> Dict[str, Any]:
        if not hasattr(lora, "project_active_adapter_gradients"):
            return {"hook_available": False, "projected": False}
        active = lora.get_active_adapter_name()
        return lora.project_active_adapter_gradients(
            reference_adapters=self._reference_adapters(active),
            strength=self.projection_strength,
        )


def _method_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    advanced_cfg = cfg.get("advanced_baseline", {}) if isinstance(cfg.get("advanced_baseline", {}), dict) else {}
    method_cfg = advanced_cfg.get(name, {}) if isinstance(advanced_cfg.get(name, {}), dict) else {}
    return method_cfg


def _to_pairs(examples: List[Example]) -> Tuple[List[Tuple[str, str]], List[str]]:
    pairs: List[Tuple[str, str]] = []
    targets: List[str] = []
    for ex in examples:
        pairs.append((ex.instruction, ex.input))
        targets.append(ex.output)
    return pairs, targets


def _batch(pairs: List[Tuple[str, str]], targets: List[str], batch_size: int):
    bs = max(1, int(batch_size))
    for i in range(0, len(pairs), bs):
        yield pairs[i : i + bs], targets[i : i + bs]
