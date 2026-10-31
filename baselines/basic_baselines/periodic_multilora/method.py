from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.data import Example, Segment


class PeriodicMultiLoRAMethod:
    """
    Baseline C: Periodic MultiLoRA

    - Multiple LoRA branches
    - Spawn a new branch every N segments (fixed schedule)
    - No drift detector, no router
    - Inference uses the latest branch (simplified)
    """

    name = "periodic_multilora"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        per_cfg = cfg.get("periodic", {})
        self.spawn_every = int(per_cfg.get("spawn_every_segments", 2))
        self.max_branches = int(per_cfg.get("max_branches", 8))
        self._branch_names: List[str] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        if not self._branch_names:
            # first branch
            if "b0" not in lora.list_adapters():
                lora.create_adapter("b0")
            lora.set_active_adapter("b0")
            self._branch_names = ["b0"]
        else:
            if self.spawn_every > 0 and (segment.segment_id % self.spawn_every == 0):
                name = f"b{len(self._branch_names)}"
                if len(self._branch_names) < self.max_branches:
                    if name not in lora.list_adapters():
                        lora.create_adapter(name)
                    lora.set_active_adapter(name)
                    self._branch_names.append(name)
                else:
                    # when max reached, keep training latest
                    lora.set_active_adapter(self._branch_names[-1])
            else:
                # keep latest
                lora.set_active_adapter(self._branch_names[-1])

        return {"active_adapter": lora.get_active_adapter_name(), "num_branches": len(self._branch_names)}

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
        batch_accs = []

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                lora.step_adapter()
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                metrics["batches"] += 1

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["num_branches"] = len(self._branch_names)
        return metrics

    def get_inference_adapter(self) -> str:
        return self._branch_names[-1] if self._branch_names else "default"


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

