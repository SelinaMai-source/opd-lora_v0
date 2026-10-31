from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.data import Example, Segment


class ProgressivePromptsMethod:
    """
    Scaffold for Progressive Prompts.

    The current project has no soft-prompt parameter module yet. This class
    records the prompt schedule and runs through the unified train/eval loop
    using the active adapter as a placeholder update path.
    """

    name = "progressive_prompts"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        method_cfg = _method_cfg(cfg, "progressive_prompts")
        self.prompt_prefix = str(method_cfg.get("prompt_prefix", "pp_s"))
        self.prompt_length = int(method_cfg.get("prompt_length", 20))
        self.freeze_previous_prompts = bool(method_cfg.get("freeze_previous_prompts", True))
        self._prompt_names: List[str] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        prompt_name = f"{self.prompt_prefix}{segment.segment_id}"
        if prompt_name not in self._prompt_names:
            self._prompt_names.append(prompt_name)
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {
            "active_adapter": lora.get_active_adapter_name(),
            "active_prompt": prompt_name,
            "num_prompts": len(self._prompt_names),
            "prompt_length": self.prompt_length,
            "scaffold_status": "soft_prompt_module_not_implemented",
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

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                lora.step_adapter()
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                batch_losses.append(float(out.get("train_loss", 0.0)))
                metrics["batches"] += 1

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["num_prompts"] = len(self._prompt_names)
        metrics["prompt_length"] = self.prompt_length
        metrics["scaffold_status"] = "smoke_test_ready_no_soft_prompt_training"
        return metrics


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
