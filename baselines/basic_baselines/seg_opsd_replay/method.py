from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from baselines.basic_baselines.seg_opsd.method import SegOPSDMethod
from core.data import Segment


@dataclass
class ReplayItem:
    instruction: str
    input_text: str
    target: str


class SegOPSDReplayMethod(SegOPSDMethod):
    """Seg OPSD with 80% current-segment / 20% replay-buffer mix on every SFT and OPSD block."""

    name = "seg_opsd_replay"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        replay_cfg = cfg.get("replay", {}) if isinstance(cfg.get("replay", {}), dict) else {}
        self.buffer_size = int(replay_cfg.get("buffer_size", 2048))
        self.replay_ratio = float(replay_cfg.get("replay_ratio", 0.2))
        self.strategy = str(replay_cfg.get("strategy", "uniform"))
        self._buffer: List[ReplayItem] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        info = super().on_segment_start(segment=segment, model=model, lora=lora)
        info["replay_buffer_size"] = len(self._buffer)
        info["replay_ratio"] = float(self.replay_ratio)
        return info

    def _iter_epoch_pairs(
        self, current_pairs: List[Tuple[str, str]], current_targets: List[str]
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        return self._mix_current_and_replay(current_pairs, current_targets)

    def _after_segment(self, current_pairs: List[Tuple[str, str]], current_targets: List[str]) -> None:
        self._add_to_buffer(current_pairs, current_targets)

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
        # Target mix: replay_ratio of the mixed set (0.2 → 80% current / 20% replay).
        n_cur = len(pairs)
        n_replay = int(round(n_cur * self.replay_ratio / max(1e-8, 1.0 - self.replay_ratio)))
        n_replay = max(0, min(n_replay, len(self._buffer)))
        if n_replay == 0:
            return pairs, targets
        replay_items = random.sample(self._buffer, k=n_replay)
        mixed_pairs = list(pairs) + [(it.instruction, it.input_text) for it in replay_items]
        mixed_targets = list(targets) + [it.target for it in replay_items]
        combined = list(zip(mixed_pairs, mixed_targets))
        random.shuffle(combined)
        return [p for p, _ in combined], [t for _, t in combined]
