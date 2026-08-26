from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from baselines.basic_baselines.opsd.method import _batch, _to_pairs, run_opsd_step
from core.data import Segment


@dataclass
class ReplayItem:
    instruction: str
    input_text: str
    target: str


class SegOPSDMethod:
    """
    Within each CL segment, alternate SFT (fit_batch) and OPSD every K optimizer steps.

    After each completed SFT block, copy `default` LoRA weights into a frozen `teacher`
    adapter; OPSD then uses that frozen LoRA (not disable_adapter / base).
    """

    name = "seg_opsd"
    TEACHER_ADAPTER = "teacher"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        opsd_cfg = cfg.get("opsd", {}) if isinstance(cfg.get("opsd", {}), dict) else {}
        seg_cfg = cfg.get("seg_opsd", {}) if isinstance(cfg.get("seg_opsd", {}), dict) else {}
        replay_cfg = cfg.get("replay", {}) if isinstance(cfg.get("replay", {}), dict) else {}

        self.k = int(seg_cfg.get("k", 25))
        if self.k < 1:
            raise ValueError(f"seg_opsd.k must be >= 1, got: {self.k}")
        self.teacher_adapter = str(seg_cfg.get("teacher_adapter", self.TEACHER_ADAPTER)).strip() or self.TEACHER_ADAPTER

        self.divergence = str(opsd_cfg.get("divergence", "jsd")).strip().lower()
        if self.divergence not in {"jsd", "kl"}:
            raise ValueError(f"opsd.divergence must be 'jsd' or 'kl', got: {self.divergence}")
        self.beta = float(opsd_cfg.get("beta", 0.5))
        self.rollout_temperature = float(opsd_cfg.get("rollout_temperature", 1.0))
        self.rollout_top_p = float(opsd_cfg.get("rollout_top_p", 1.0))
        self.rollout_max_new_tokens = int(opsd_cfg.get("rollout_max_new_tokens", 0))
        self.teacher_format_constraint = bool(opsd_cfg.get("teacher_format_constraint", False))

        self.replay_ratio = float(replay_cfg.get("replay_ratio", 0.0))
        self.buffer_size = int(replay_cfg.get("buffer_size", 2048))
        self._buffer: List[ReplayItem] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ensure_teacher_adapter(lora, self.teacher_adapter)
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {
            "active_adapter": lora.get_active_adapter_name(),
            "teacher_adapter": self.teacher_adapter,
            "seg_k": int(self.k),
            "replay_buffer_size": len(self._buffer),
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
        _ensure_teacher_adapter(lora, self.teacher_adapter)
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")

        metrics: Dict[str, Any] = {
            "batches": 0,
            "sft_steps": 0,
            "opsd_steps": 0,
            "sft_blocks_completed": 0,
            "opsd_blocks_completed": 0,
        }
        mode = "sft"
        steps_in_block = 0
        batch_accs: List[float] = []
        batch_losses: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        rollout_token_total = 0
        rollout_count = 0

        for _ in range(max(1, epochs)):
            pairs, targets = self._mix_current_and_replay(current_pairs, current_targets)
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                if mode == "sft":
                    out = model.fit_batch(b_pairs, b_targets, lr=lr)
                    step_stats = lora.step_adapter()
                    batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                    batch_losses.append(float(out.get("train_loss", 0.0)))
                    grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                    lr_values.append(float(step_stats.get("lr", lr)))
                    metrics["batches"] += 1
                    metrics["sft_steps"] += 1
                    steps_in_block += 1
                    if steps_in_block >= self.k:
                        lora.copy_adapter_weights("default", self.teacher_adapter)
                        metrics["sft_blocks_completed"] += 1
                        mode = "opsd"
                        steps_in_block = 0
                else:
                    step = run_opsd_step(
                        model=model,
                        lora=lora,
                        b_pairs=b_pairs,
                        b_targets=b_targets,
                        lr=lr,
                        divergence=self.divergence,
                        beta=self.beta,
                        rollout_temperature=self.rollout_temperature,
                        rollout_top_p=self.rollout_top_p,
                        rollout_max_new_tokens=self.rollout_max_new_tokens,
                        teacher_source="adapter",
                        teacher_adapter=self.teacher_adapter,
                        teacher_format_constraint=self.teacher_format_constraint,
                    )
                    if step.get("skipped"):
                        continue
                    batch_losses.append(float(step["loss"]))
                    grad_norms.append(float(step["grad_norm"]))
                    lr_values.append(float(step["lr"]))
                    rollout_token_total += int(step["rollout_token_total"])
                    rollout_count += int(step["rollout_count"])
                    metrics["batches"] += 1
                    metrics["opsd_steps"] += 1
                    steps_in_block += 1
                    if steps_in_block >= self.k:
                        metrics["opsd_blocks_completed"] += 1
                        mode = "sft"
                        steps_in_block = 0

        if self.replay_ratio > 0:
            self._add_to_buffer(current_pairs, current_targets)

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["rollout_mean_tokens"] = rollout_token_total / max(1, rollout_count)
        metrics["seg_k"] = int(self.k)
        metrics["teacher_adapter"] = self.teacher_adapter
        metrics["replay_ratio"] = float(self.replay_ratio)
        metrics["replay_buffer_size"] = len(self._buffer)
        metrics["active_adapter"] = lora.get_active_adapter_name()
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
        # Mix so replay is ~replay_ratio of the combined set (80/20 when ratio=0.2).
        n_replay = int(round(len(pairs) * self.replay_ratio / max(1e-8, 1.0 - self.replay_ratio)))
        n_replay = max(0, min(n_replay, len(self._buffer)))
        if n_replay == 0:
            return pairs, targets
        replay_items = random.sample(self._buffer, k=n_replay)
        mixed = list(zip(pairs, targets)) + [
            ((it.instruction, it.input_text), it.target) for it in replay_items
        ]
        random.shuffle(mixed)
        mixed_pairs = [p for p, _ in mixed]
        mixed_targets = [t for _, t in mixed]
        return mixed_pairs, mixed_targets


class SegOPSDReplayMethod(SegOPSDMethod):
    """Seg-OPSD with 80% current / 20% replay from previously seen segments."""

    name = "seg_opsd_replay"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        if self.replay_ratio <= 0:
            self.replay_ratio = 0.2


def _ensure_teacher_adapter(lora: Any, name: str) -> None:
    if name in lora.list_adapters():
        if hasattr(lora, "freeze_adapter"):
            lora.freeze_adapter(name)
        return
    lora.create_adapter(name)
    lora.freeze_adapter(name)
    if "default" in lora.list_adapters():
        lora.set_active_adapter("default")
