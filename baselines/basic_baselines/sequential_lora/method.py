from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.data import Example, Segment


class SequentialLoRAMethod:
    """
    Baseline A: Sequential LoRA

    - Single LoRA branch (no bank, no router)
    - Train segment by segment, always on the same adapter
    """

    name = "sequential_lora"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        # Ensure using default adapter for sequential baseline.
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {"active_adapter": lora.get_active_adapter_name()}

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
        metrics: Dict[str, Any] = {"batches": 0, "mean_batch_acc": 0.0}
        pairs, targets = _to_pairs(segment.train)

        batch_accs = []
        batch_losses = []
        answer_token_accs = []
        grad_norms = []
        lr_values = []
        total_tokens = 0
        supervised_tokens = 0
        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                step_stats = lora.step_adapter()
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                batch_losses.append(float(out.get("train_loss", 0.0)))
                answer_token_accs.append(float(out.get("train_answer_token_acc", 0.0)))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                total_tokens += int(out.get("num_total_tokens", 0))
                supervised_tokens += int(out.get("num_supervised_tokens", 0))
                metrics["batches"] += 1

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["train.answer_token_acc"] = sum(answer_token_accs) / max(1, len(answer_token_accs))
        metrics["num_total_tokens"] = int(total_tokens)
        metrics["num_supervised_tokens"] = int(supervised_tokens)
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics


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

