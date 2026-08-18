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


class ContinualT0Method:
    """
    Scaffold for Continual-T0 style instruction rehearsal.

    This implements the minimal rehearsal wiring needed by the unified baseline
    loop. It does not reproduce the original T0/T5 checkpoint or task mixture.
    """

    name = "continual_t0"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        method_cfg = _method_cfg(cfg, "continual_t0")
        replay_cfg = cfg.get("replay", {}) if isinstance(cfg.get("replay", {}), dict) else {}
        self.buffer_size = int(method_cfg.get("buffer_size", replay_cfg.get("buffer_size", 256)))
        self.replay_ratio = float(method_cfg.get("replay_ratio", replay_cfg.get("replay_ratio", 0.01)))
        self.strategy = str(method_cfg.get("strategy", replay_cfg.get("strategy", "uniform")))
        self._buffer: List[ReplayItem] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {
            "active_adapter": lora.get_active_adapter_name(),
            "replay_buffer_size": len(self._buffer),
            "replay_ratio": self.replay_ratio,
            "scaffold_status": "t0_checkpoint_and_mixture_not_implemented",
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
        current_pairs, current_targets = _to_pairs(segment.train)
        mix_pairs, mix_targets = self._mix_current_and_replay(current_pairs, current_targets)
        self._add_to_buffer(current_pairs, current_targets)

        metrics: Dict[str, Any] = {"batches": 0, "mean_batch_acc": 0.0}
        batch_accs: List[float] = []
        batch_losses: List[float] = []

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(mix_pairs, mix_targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                lora.step_adapter()
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                batch_losses.append(float(out.get("train_loss", 0.0)))
                metrics["batches"] += 1

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["replay_buffer_size"] = len(self._buffer)
        metrics["replay_ratio"] = self.replay_ratio
        metrics["scaffold_status"] = "smoke_test_ready_instruction_replay_only"
        return metrics

    def _add_to_buffer(self, pairs: List[Tuple[str, str]], targets: List[str]) -> None:
        for (instruction, input_text), y in zip(pairs, targets):
            self._buffer.append(ReplayItem(instruction=instruction, input_text=input_text, target=y))
        if len(self._buffer) > self.buffer_size:
            overflow = len(self._buffer) - self.buffer_size
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
        return pairs + replay_pairs, targets + replay_targets


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
