from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.data import Example, Segment


class LBCLMethod:
    """
    Scaffold for LB-CL.

    This keeps the unified baseline loop runnable while exposing the key LB-CL
    state transitions: cache old low-rank summaries, initialize the new segment
    from selected prior summaries, then train with a verifiable projection hook.
    The summary is compact SVD triplets over LoRA matrices or debug adapter
    vectors; full paper-level triplet injection remains a follow-up.
    """

    name = "lb_cl"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        method_cfg = _method_cfg(cfg, "lb_cl")
        self.adapter_prefix = str(method_cfg.get("adapter_prefix", "lbcl_s"))
        self.top_triplets = int(method_cfg.get("top_triplets", 8))
        self.projection_strength = float(method_cfg.get("projection_strength", 1.0))
        self.freeze_previous_adapters = bool(method_cfg.get("freeze_previous_adapters", True))
        self._adapter_names: List[str] = []
        self._low_rank_cache: List[Dict[str, Any]] = []

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
            "cached_low_rank_summaries": len(self._low_rank_cache),
            "top_triplets": self.top_triplets,
            "scaffold_status": "svd_sensitivity_and_gradient_projection_not_implemented",
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
        projection_calls = 0
        projected_calls = 0

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                projection_stats = self._project_gradients(lora)
                projection_calls += int(bool(projection_stats.get("hook_available", False)))
                projected_calls += int(bool(projection_stats.get("projected", False)))
                step_stats = lora.step_adapter()
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                batch_losses.append(float(out.get("train_loss", 0.0)))
                metrics["batches"] += 1

        active_adapter = lora.get_active_adapter_name()
        svd_summary = self._summarize_adapter(lora, active_adapter)
        self._low_rank_cache.append({"segment_id": segment.segment_id, **svd_summary})
        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["active_adapter"] = active_adapter
        metrics["num_adapters"] = len(self._adapter_names)
        metrics["cached_low_rank_summaries"] = len(self._low_rank_cache)
        metrics["svd_summary_num_matrices"] = int(svd_summary.get("num_matrices", 0))
        metrics["svd_top_triplets_recorded"] = len(svd_summary.get("top_triplets", []))
        metrics["gradient_projection_hook_calls"] = int(projection_calls)
        metrics["gradient_projection_applied_calls"] = int(projected_calls)
        metrics["projection_strength"] = self.projection_strength
        metrics["grad_norm"] = float(step_stats.get("grad_norm", 0.0)) if "step_stats" in locals() else 0.0
        metrics["scaffold_status"] = "smoke_test_ready_with_svd_and_projection_hook"
        return metrics

    def _reference_adapters(self, active_adapter: str) -> List[str]:
        return [row["adapter"] for row in self._low_rank_cache if row.get("adapter") != active_adapter]

    def _project_gradients(self, lora: Any) -> Dict[str, Any]:
        if not hasattr(lora, "project_active_adapter_gradients"):
            return {"hook_available": False, "projected": False}
        active = lora.get_active_adapter_name()
        return lora.project_active_adapter_gradients(
            reference_adapters=self._reference_adapters(active),
            strength=self.projection_strength,
        )

    def _summarize_adapter(self, lora: Any, adapter_name: str) -> Dict[str, Any]:
        if hasattr(lora, "summarize_adapter_svd"):
            summary = lora.summarize_adapter_svd(adapter_name, top_k=self.top_triplets)
            if isinstance(summary, dict):
                return summary
        return {"adapter": adapter_name, "num_matrices": 0, "top_triplets": []}


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
