from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from core.data import Example, Segment


@dataclass
class ReplayItem:
    instruction: str
    input_text: str
    target: str


class ReplayLoRAMethod:
    """
    Baseline B: Replay LoRA

    - Single LoRA branch
    - Maintain a replay buffer across segments
    - Train each segment with a mixture of current + replay examples
    """

    name = "replay_lora"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        replay_cfg = cfg.get("replay", {})
        self.buffer_size = int(replay_cfg.get("buffer_size", 256))
        self.replay_ratio = float(replay_cfg.get("replay_ratio", 0.3))
        self.strategy = str(replay_cfg.get("strategy", "uniform"))
        self._buffer: List[ReplayItem] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {"active_adapter": lora.get_active_adapter_name(), "replay_buffer_size": len(self._buffer)}

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
        current_pairs, current_targets = _to_pairs(segment.train)
        self._add_to_buffer(current_pairs, current_targets)

        metrics: Dict[str, Any] = {"batches": 0, "mean_batch_acc": 0.0}
        batch_accs = []

        for _ in range(max(1, epochs)):
            # Build a mixed training set for this epoch
            mix_pairs, mix_targets = self._mix_current_and_replay(current_pairs, current_targets)
            for b_pairs, b_targets in _batch(mix_pairs, mix_targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                lora.step_adapter()
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                metrics["batches"] += 1

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["replay_buffer_size"] = len(self._buffer)
        return metrics

    def _add_to_buffer(self, pairs: List[Tuple[str, str]], targets: List[str]) -> None:
        for (instruction, input_text), y in zip(pairs, targets):
            self._buffer.append(ReplayItem(instruction=instruction, input_text=input_text, target=y))
        # keep buffer size
        if len(self._buffer) > self.buffer_size:
            overflow = len(self._buffer) - self.buffer_size
            # simple policy: drop oldest
            self._buffer = self._buffer[overflow:]

    def _mix_current_and_replay(
        self, pairs: List[Tuple[str, str]], targets: List[str]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        if not self._buffer or self.replay_ratio <= 0:
            return pairs, targets

        num_replay = int(round(len(pairs) * self.replay_ratio))
        num_replay = max(0, min(num_replay, len(self._buffer)))

        if num_replay == 0:
            return pairs, targets

        replay_items = random.sample(self._buffer, k=num_replay)
        replay_pairs = [(it.instruction, it.input_text) for it in replay_items]
        replay_targets = [it.target for it in replay_items]

        mixed_pairs = pairs + replay_pairs
        mixed_targets = targets + replay_targets
        return mixed_pairs, mixed_targets


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

