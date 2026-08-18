from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Dict, List, Optional

from core.data import ContinualStream, Example


@dataclass
class AnchorItem:
    anchor_id: str
    segment_id: int
    segment_name: str
    instruction: str
    input_text: str
    output: str
    split: str
    complexity: float


@dataclass
class AnchorSet:
    core: List[AnchorItem]
    probe: List[AnchorItem]

    def summary(self) -> Dict[str, Any]:
        return {
            "num_core": len(self.core),
            "num_probe": len(self.probe),
            "num_total": len(self.core) + len(self.probe),
            "core_segment_ids": sorted({int(x.segment_id) for x in self.core}),
            "probe_segment_ids": sorted({int(x.segment_id) for x in self.probe}),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "core": [asdict(x) for x in self.core],
            "probe": [asdict(x) for x in self.probe],
        }


@dataclass
class DriftEvent:
    triggered: bool
    calibrated: bool
    score: float
    threshold: float
    reason: str
    segment_id: int
    monitor_step: int
    core_mean_nll: float
    probe_mean_nll: float
    core_dev: float
    probe_dev: float
    core_cusum: float
    probe_cusum: float
    core_baseline: float
    probe_baseline: float
    core_slack: float
    probe_slack: float
    consecutive_probe_hits: int

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


def build_anchor_set(stream: ContinualStream, cfg: Dict[str, Any], *, seed: int, model: Optional[Any] = None) -> AnchorSet:
    anchor_size = max(2, int(cfg.get("anchor_size", 64)))
    core_fraction = float(cfg.get("anchor_core_fraction", 0.5))
    curriculum_strategy = str(cfg.get("curriculum_strategy", "easy_core_hard_probe")).strip()
    # "train" avoids any use of held-out eval examples for drift monitoring
    # (protocol hygiene); "eval" is the legacy behaviour.
    anchor_source = str(cfg.get("anchor_source", "eval")).strip() or "eval"
    if anchor_source not in {"train", "eval"}:
        raise ValueError("drift.anchor_source must be one of: train | eval")

    def _segment_examples(seg: Any) -> List[Example]:
        return list(seg.train) if anchor_source == "train" else list(seg.eval)

    rng = random.Random(int(seed))

    all_items: List[AnchorItem] = []
    for seg in stream.stream:
        for idx, ex in enumerate(_segment_examples(seg)):
            all_items.append(
                AnchorItem(
                    anchor_id=f"s{seg.segment_id:03d}_e{idx:04d}",
                    segment_id=int(seg.segment_id),
                    segment_name=str(seg.segment_name),
                    instruction=str(ex.instruction),
                    input_text=str(ex.input),
                    output=str(ex.output),
                    split="unassigned",
                    complexity=0.0,
                )
            )

    selected: List[AnchorItem] = []
    if model is not None and hasattr(model, "get_activations_tensor") and len(all_items) > anchor_size:
        import torch
        from core.formatting import format_for_infer
        tok = getattr(model, "tokenizer", None)
        prompts = []
        for item in all_items:
            if tok is not None:
                prompts.append(format_for_infer(tok, item.instruction, item.input_text, add_generation_prompt=True))
            else:
                prompts.append(f"{item.instruction}\n\n{item.input_text}")
        
        with torch.no_grad():
            features = model.get_activations_tensor(prompts, with_grad=False)
            
        # K-Center Greedy
        selected_indices = [rng.randint(0, len(all_items) - 1)]
        min_distances = torch.cdist(features, features[selected_indices[0]:selected_indices[0]+1], p=2).squeeze(1)
        
        while len(selected_indices) < anchor_size:
            next_idx = int(torch.argmax(min_distances).item())
            selected_indices.append(next_idx)
            dist_to_new = torch.cdist(features, features[next_idx:next_idx+1], p=2).squeeze(1)
            min_distances = torch.minimum(min_distances, dist_to_new)
            
        for idx in selected_indices:
            item = all_items[idx]
            item.complexity = _anchor_complexity(item)
            selected.append(item)
    else:
        # Fallback to random bucket approach
        buckets: List[List[AnchorItem]] = []
        for seg in stream.stream:
            examples = _segment_examples(seg)
            rng.shuffle(examples)
            bucket: List[AnchorItem] = []
            for idx, ex in enumerate(examples):
                bucket.append(
                    AnchorItem(
                        anchor_id=f"s{seg.segment_id:03d}_e{idx:04d}",
                        segment_id=int(seg.segment_id),
                        segment_name=str(seg.segment_name),
                        instruction=str(ex.instruction),
                        input_text=str(ex.input),
                        output=str(ex.output),
                        split="unassigned",
                        complexity=0.0,
                    )
                )
            buckets.append(bucket)

        while len(selected) < anchor_size:
            progressed = False
            for bucket in buckets:
                if bucket and len(selected) < anchor_size:
                    item = bucket.pop(0)
                    item.complexity = _anchor_complexity(item)
                    selected.append(item)
                    progressed = True
            if not progressed:
                break

    if len(selected) < 2:
        raise ValueError("Need at least two anchor examples to build core/probe sets.")

    ordered = _order_anchors_for_curriculum(selected, strategy=curriculum_strategy)
    core_size = int(round(len(ordered) * core_fraction))
    core_size = max(1, min(len(ordered) - 1, core_size))
    probe_size = len(ordered) - core_size
    if probe_size <= 0:
        probe_size = 1
        core_size = len(ordered) - 1

    core_items: List[AnchorItem] = []
    probe_items: List[AnchorItem] = []
    for idx, item in enumerate(ordered):
        clone = AnchorItem(**asdict(item))
        if idx < core_size:
            clone.split = "core"
            core_items.append(clone)
        else:
            clone.split = "probe"
            probe_items.append(clone)
    return AnchorSet(core=core_items, probe=probe_items)


def _order_anchors_for_curriculum(items: List[AnchorItem], *, strategy: str) -> List[AnchorItem]:
    ordered = sorted(items, key=lambda x: (x.complexity, x.anchor_id))
    if strategy in {"easy_core_hard_probe", "curriculum", "easy_to_hard"}:
        return ordered
    if strategy in {"hard_core_easy_probe", "reverse_curriculum", "hard_to_easy"}:
        return list(reversed(ordered))
    if strategy in {"interleaved", "balanced"}:
        easy = ordered[::2]
        hard = ordered[1::2]
        return easy + hard
    raise ValueError(
        "drift.curriculum_strategy must be one of: "
        "easy_core_hard_probe | hard_core_easy_probe | interleaved"
    )


def _anchor_complexity(item: AnchorItem) -> float:
    text = f"{item.instruction}\n{item.input_text}"
    punct = sum(1 for ch in text if ch in ":;,.!?[]{}()<>/\\")
    newlines = text.count("\n")
    digits = sum(1 for ch in text if ch.isdigit())
    uppercase = sum(1 for ch in text if ch.isupper())
    return float(len(text) + 6 * punct + 12 * newlines + 4 * digits + 2 * uppercase)


class DriftDetector:
    """
    Drift detector backed by two-tier anchor monitoring and calibrated CUSUM.

    Design:
      - Monitor a fixed anchor set split into `core` and `probe`
      - Track teacher-forced answer NLL on both subsets
      - Calibrate on an initial window, then apply one-sided CUSUM to positive deviation
      - Trigger only when probe degradation is sustained and core also shows consistent shift
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.anchor_size = int(cfg.get("anchor_size", 64))
        self.anchor_core_fraction = float(cfg.get("anchor_core_fraction", 0.5))
        self.monitor_interval = int(cfg.get("monitor_interval", 50))
        self.calibration_window = int(cfg.get("calibration_window", 1))
        self.threshold = float(cfg.get("threshold", 0.01)) 
        self.score_ema = float(cfg.get("score_ema", 0.9))
        self.core_threshold_scale = float(cfg.get("core_threshold_scale", 1.1)) 
        self.probe_threshold_scale = float(cfg.get("probe_threshold_scale", 1.1)) 
        self.core_slack_scale = float(cfg.get("core_slack_scale", 0.1)) 
        self.probe_slack_scale = float(cfg.get("probe_slack_scale", 0.1)) 
        self.core_guard_scale = float(cfg.get("core_guard_scale", 0.5))
        self.min_consecutive_probe_hits = int(cfg.get("min_consecutive_probe_hits", 1))
        self.shift_stat = str(cfg.get("shift_stat", "degradation")).strip() or "degradation"
        if self.shift_stat not in {"degradation", "absolute"}:
            raise ValueError("drift.shift_stat must be one of: degradation | absolute")
        # If True, spawning is forced at every segment boundary. This leaks
        # ground-truth task-boundary information and must stay False for any
        # task-agnostic (fair) experiment; kept only for ablation/debugging.
        self.force_spawn_on_segment_boundary = bool(cfg.get("force_spawn_on_segment_boundary", False))
        self.meta_threshold_enabled = bool(cfg.get("meta_threshold_enabled", False))
        self.meta_threshold_scale = float(cfg.get("meta_threshold_scale", 1.0))
        self.meta_threshold_min = float(cfg.get("meta_threshold_min", self.threshold))
        self.meta_threshold_max = float(cfg.get("meta_threshold_max", max(self.threshold, 10.0)))

        self._num_updates = 0
        self._history: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []

        self._core_ema: Optional[float] = None
        self._probe_ema: Optional[float] = None
        self._core_calibration: List[float] = []
        self._probe_calibration: List[float] = []
        self._core_baseline: Optional[float] = None
        self._probe_baseline: Optional[float] = None
        self._core_std: Optional[float] = None
        self._probe_std: Optional[float] = None
        self._core_cusum: float = 0.0
        self._probe_cusum: float = 0.0
        self._consecutive_probe_hits: int = 0

    def reset(self, *, keep_history: bool = False) -> None:
        self._num_updates = 0
        if not keep_history:
            self._history = []
            self._events = []
        self._core_ema = None
        self._probe_ema = None
        self._core_calibration = []
        self._probe_calibration = []
        self._core_baseline = None
        self._probe_baseline = None
        self._core_std = None
        self._probe_std = None
        self._core_cusum = 0.0
        self._probe_cusum = 0.0
        self._consecutive_probe_hits = 0

    def update(
        self,
        *,
        core_mean_nll: float,
        probe_mean_nll: float,
        segment_id: int,
        monitor_step: Optional[int] = None,
    ) -> DriftEvent:
        self._num_updates += 1
        if self._core_ema is None:
            self._core_ema = float(core_mean_nll)
            self._probe_ema = float(probe_mean_nll)
        else:
            self._core_ema = self.score_ema * self._core_ema + (1.0 - self.score_ema) * float(core_mean_nll)
            self._probe_ema = self.score_ema * self._probe_ema + (1.0 - self.score_ema) * float(probe_mean_nll)

        core_obs = float(core_mean_nll)
        probe_obs = float(probe_mean_nll)
        step_id = int(monitor_step if monitor_step is not None else self._num_updates)

        if not hasattr(self, '_last_seen_segment_id'):
            self._last_seen_segment_id = segment_id
            
        force_trigger = False
        if segment_id != self._last_seen_segment_id:
            # Only force-trigger when explicitly enabled (unfair / ablation-only mode).
            if self.force_spawn_on_segment_boundary:
                force_trigger = True
            self._last_seen_segment_id = segment_id

        if self._core_baseline is None or self._probe_baseline is None:
            self._core_calibration.append(core_obs)
            self._probe_calibration.append(probe_obs)
            if len(self._core_calibration) >= max(1, self.calibration_window):
                self._core_baseline = _mean(self._core_calibration)
                self._probe_baseline = _mean(self._probe_calibration)
                self._core_std = _std(self._core_calibration)
                self._probe_std = _std(self._probe_calibration)
                reason = "calibration_complete"
            else:
                reason = "calibrating"
            event = DriftEvent(
                triggered=False,
                calibrated=self._core_baseline is not None and self._probe_baseline is not None,
                score=0.0,
                threshold=self.threshold,
                reason=reason,
                segment_id=int(segment_id),
                monitor_step=step_id,
                core_mean_nll=core_obs,
                probe_mean_nll=probe_obs,
                core_dev=0.0,
                probe_dev=0.0,
                core_cusum=float(self._core_cusum),
                probe_cusum=float(self._probe_cusum),
                core_baseline=float(self._core_baseline or core_obs),
                probe_baseline=float(self._probe_baseline or probe_obs),
                core_slack=0.0,
                probe_slack=0.0,
                consecutive_probe_hits=int(self._consecutive_probe_hits),
            )
            self._history.append(event.to_row())
            return event

        core_std = float(self._core_std or 1e-6)
        probe_std = float(self._probe_std or 1e-6)
        core_baseline = float(self._core_baseline)
        probe_baseline = float(self._probe_baseline)
        raw_core_dev = core_obs - core_baseline
        raw_probe_dev = probe_obs - probe_baseline
        if self.shift_stat == "absolute":
            core_dev = abs(raw_core_dev)
            probe_dev = abs(raw_probe_dev)
        else:
            core_dev = raw_core_dev
            probe_dev = raw_probe_dev
        core_slack = max(1e-6, core_std * self.core_slack_scale)
        probe_slack = max(1e-6, probe_std * self.probe_slack_scale)
        core_threshold = max(self.threshold, core_std * self.core_threshold_scale)
        probe_threshold = max(self.threshold, probe_std * self.probe_threshold_scale)
        
        # Make the detector much more sensitive to ensure branching
        probe_threshold = min(probe_threshold, self.threshold * 1.5)
        core_threshold = min(core_threshold, self.threshold * 1.5)
        if self.meta_threshold_enabled:
            baseline_gap = abs(probe_baseline - core_baseline)
            core_threshold = _clamp(
                self.threshold + self.meta_threshold_scale * (core_std + baseline_gap),
                self.meta_threshold_min,
                self.meta_threshold_max,
            )
            probe_threshold = _clamp(
                self.threshold + self.meta_threshold_scale * (probe_std + baseline_gap),
                self.meta_threshold_min,
                self.meta_threshold_max,
            )

        self._core_cusum = max(0.0, self._core_cusum + core_dev - core_slack)
        self._probe_cusum = max(0.0, self._probe_cusum + probe_dev - probe_slack)

        probe_hit = (self._probe_cusum >= probe_threshold) or (probe_dev >= probe_threshold)
        if probe_hit:
            self._consecutive_probe_hits += 1
        else:
            self._consecutive_probe_hits = 0

        core_guard = True
        triggered = bool(core_guard and self._consecutive_probe_hits >= max(1, self.min_consecutive_probe_hits))
        reason = "stable"
        if probe_hit and not core_guard:
            reason = "probe_shift_without_core_guard"
        elif triggered:
            reason = "probe_cusum_and_core_guard"
            
        if force_trigger:
            triggered = True
            reason = "thermodynamic_capacity_or_segment_boundary"

        event = DriftEvent(
            triggered=triggered,
            calibrated=True,
            score=float(self._probe_cusum),
            threshold=float(probe_threshold),
            reason=reason,
            segment_id=int(segment_id),
            monitor_step=step_id,
            core_mean_nll=core_obs,
            probe_mean_nll=probe_obs,
            core_dev=float(core_dev),
            probe_dev=float(probe_dev),
            core_cusum=float(self._core_cusum),
            probe_cusum=float(self._probe_cusum),
            core_baseline=float(core_baseline),
            probe_baseline=float(probe_baseline),
            core_slack=float(core_slack),
            probe_slack=float(probe_slack),
            consecutive_probe_hits=int(self._consecutive_probe_hits),
        )
        row = event.to_row()
        row["raw_core_dev"] = float(raw_core_dev)
        row["raw_probe_dev"] = float(raw_probe_dev)
        row["shift_stat"] = self.shift_stat
        row["core_threshold"] = float(core_threshold)
        row["probe_threshold"] = float(probe_threshold)
        row["meta_threshold_enabled"] = bool(self.meta_threshold_enabled)
        self._history.append(row)
        if triggered:
            self._events.append(row)
            self._core_cusum = 0.0
            self._probe_cusum = 0.0
            self._consecutive_probe_hits = 0
        return event

    def state_dict(self) -> Dict[str, Any]:
        return {
            "num_updates": self._num_updates,
            "anchor_size": int(self.anchor_size),
            "anchor_core_fraction": float(self.anchor_core_fraction),
            "monitor_interval": int(self.monitor_interval),
            "calibration_window": int(self.calibration_window),
            "threshold": self.threshold,
            "score_ema": self.score_ema,
            "core_baseline": self._core_baseline,
            "probe_baseline": self._probe_baseline,
            "core_std": self._core_std,
            "probe_std": self._probe_std,
            "core_ema": self._core_ema,
            "probe_ema": self._probe_ema,
            "core_cusum": self._core_cusum,
            "probe_cusum": self._probe_cusum,
            "consecutive_probe_hits": self._consecutive_probe_hits,
            "shift_stat": self.shift_stat,
            "meta_threshold_enabled": bool(self.meta_threshold_enabled),
            "meta_threshold_scale": float(self.meta_threshold_scale),
            "history_tail": self._history[-50:],
            "events_tail": self._events[-50:],
        }

    def monitor_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def drift_events(self) -> List[Dict[str, Any]]:
        return list(self._events)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: List[float]) -> float:
    if len(values) <= 1:
        return 1e-6
    mu = _mean(values)
    var = sum((x - mu) ** 2 for x in values) / max(1, len(values) - 1)
    return float(max(1e-6, math.sqrt(max(0.0, var))))


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))

