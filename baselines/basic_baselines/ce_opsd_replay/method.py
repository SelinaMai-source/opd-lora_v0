from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from baselines.basic_baselines.ce_opsd.method import CeOpsdMethod, _example_golds
from core.data import Example, Segment
from core.evaluate import _score_task_aware
from core.replay_budget import (
    allocate_mixed_batch,
    cosine_similarity,
    rank_segment_ids,
    realized_replay_fraction,
    vector_l2,
)
from methods.pgdr import (
    MONITOR_EVAL_SPLIT,
    MONITOR_TRAIN_HOLDOUT,
    SELECTOR_VERSIONS,
    example_key,
    forgetting_from_accuracies,
    quota,
    score_holdout_map,
    select_forgotten,
    split_train_holdout,
)


@dataclass
class SegmentMemory:
    segment_id: int
    instruction: str
    task_score_type: str
    mean_gold_len: float
    embedding: List[float]
    examples: List[Example] = field(default_factory=list)


@dataclass
class ReplayPick:
    example: Example
    pool: str
    source_segment_id: int
    instruction: str
    input: str


_LEGACY_STRATEGIES = {"uniform", "proto"}
_UNIFIED_STRATEGIES = {"random", "compat_uniform", "true_prototype"}


class CeOpsdReplayMethod(CeOpsdMethod):
    """
    Same-step CE+OPSD with replay and optional teacher prototype injection.

    Legacy replay.strategy (v3h quota, unchanged):
      - uniform: ``quota(n, 0.2)`` random from per-segment K buffers
      - proto: ``quota(n, 0.1) + quota(n, 0.1)`` neighbor + forgotten
    replay.transfer_slot (proto 的 neighbor_ratio 槽; 默认 neighbor = v2 采样):
      - neighbor: format-compatible pool, uniform (unchanged v2). Cosine only
        orders the pool and does not change the sample. This is not true prototype.
      - anti: format-compatible lowest-cosine segment
      - uniform: all memory uniform
    replay.coverage_dedup (default false):
      - neighbor/transfer pool only: in-batch unique (instruction, input)
        and prefer low coverage count / LRU. Forgotten ranking unchanged.

    Corrected strategies share one mixed-batch ``n_replay`` from
    ``core.replay_budget`` (45 current → 11, realized 11/56). They do not call
    the legacy per-slot ``round``:
      - random: uniform over all historical memory. No similarity, no forgetting.
      - compat_uniform: Related is format-compatible uniform. Degraded is
        train-holdout forgetting. This is not a cosine prototype.
      - true_prototype: Related walks frozen instruction vectors by explicit
        L2 cosine, high to low, then samples inside that segment's K memory.
        Ties break by ascending segment_id. Degraded matches compat_uniform.

    teacher_proto.mode:
      - none / random / matched  (none == proto_replay job, do not rerun)
    """

    name = "ce_opsd_replay"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        replay_cfg = cfg.get("replay", {}) if isinstance(cfg.get("replay", {}), dict) else {}
        teacher_cfg = cfg.get("teacher_proto", {}) if isinstance(cfg.get("teacher_proto", {}), dict) else {}
        self.replay_ratio = float(replay_cfg.get("replay_ratio", 0.2))
        self.buffer_k = int(replay_cfg.get("buffer_k", replay_cfg.get("k", 10)))
        self.strategy = str(replay_cfg.get("strategy", "proto")).strip().lower()
        if self.strategy not in (_LEGACY_STRATEGIES | _UNIFIED_STRATEGIES):
            raise ValueError(
                "replay.strategy must be uniform|proto|random|compat_uniform|true_prototype, "
                f"got: {self.strategy}"
            )
        if self.strategy in _UNIFIED_STRATEGIES:
            self.budget_policy = str(replay_cfg.get("budget_policy", "unified")).strip().lower()
            self.ratio_definition = str(replay_cfg.get("ratio_definition", "mixed_batch")).strip().lower()
        else:
            # Legacy yaml files omit these keys. Do not invent a new quota for them.
            self.budget_policy = str(replay_cfg.get("budget_policy", "legacy")).strip().lower()
            self.ratio_definition = str(replay_cfg.get("ratio_definition", "")).strip().lower()
        if "target_fraction" in replay_cfg:
            self.target_fraction = float(replay_cfg.get("target_fraction"))
        else:
            self.target_fraction = float(self.replay_ratio)
        if self.strategy in _UNIFIED_STRATEGIES:
            if self.budget_policy != "unified":
                raise ValueError(
                    f"random|compat_uniform|true_prototype require budget_policy=unified, got: {self.budget_policy}"
                )
            if self.ratio_definition != "mixed_batch":
                raise ValueError(
                    f"unified selectors require ratio_definition=mixed_batch, got: {self.ratio_definition}"
                )
        self.neighbor_ratio = float(replay_cfg.get("neighbor_ratio", 0.1))
        self.forgotten_ratio = float(replay_cfg.get("forgotten_ratio", 0.1))
        self.transfer_slot = str(replay_cfg.get("transfer_slot", "neighbor")).strip().lower()
        if self.transfer_slot not in {"neighbor", "anti", "uniform"}:
            raise ValueError(
                f"replay.transfer_slot must be neighbor|anti|uniform, got: {self.transfer_slot}"
            )
        self.coverage_dedup = _as_bool(replay_cfg.get("coverage_dedup", False), False)
        self._pair_cover_n: Dict[tuple, int] = {}
        self._pair_last_tick: Dict[tuple, int] = {}
        self._cover_tick = 0
        self.teacher_proto_mode = str(teacher_cfg.get("mode", "none")).strip().lower()
        if self.teacher_proto_mode not in {"none", "random", "matched"}:
            raise ValueError(
                f"teacher_proto.mode must be none|random|matched, got: {self.teacher_proto_mode}"
            )
        self._memories: List[SegmentMemory] = []
        self._forgetting_by_segment: Dict[int, float] = {}
        self._current_memory: Optional[SegmentMemory] = None
        self._n_replay_used = 0
        self._n_neighbor_used = 0
        self._n_forgotten_used = 0
        self._n_anti_used = 0
        self._n_uniform_used = 0
        self._n_related_used = 0
        self._n_degraded_used = 0
        self._fallback_count = 0
        self._n_teacher_proto = 0
        self._last_replay_log: Dict[str, Any] = {}
        self.monitor = str(replay_cfg.get("monitor", MONITOR_EVAL_SPLIT)).strip().lower()
        if self.monitor not in SELECTOR_VERSIONS:
            raise ValueError(
                f"replay.monitor must be one of {SELECTOR_VERSIONS}, got: {self.monitor}"
            )
        self.holdout_n = int(replay_cfg.get("holdout_n", 5))
        self.holdout_seed = int(replay_cfg.get("holdout_seed", cfg.get("seed", 456)))
        self._holdouts: Dict[int, List[Example]] = {}
        self._holdout_best: Dict[int, float] = {}
        self._eval_split_consumed = False
        self._last_audit: Dict[str, Any] = {}
        self._live_metrics: Optional[Dict[str, Any]] = None
        self._current_fit: List[Example] = []
        self._current_holdout: List[Example] = []
        init_memory_path = str(replay_cfg.get("init_memory_path", "")).strip()
        if init_memory_path:
            self.load_memory_dump(init_memory_path)

    def on_eval_end(self, eval_metrics: Dict[str, Any]) -> None:
        # train_holdout must not read official-eval forgetting. The paper eval
        # still runs in core.train; this hook ignores it for selection.
        if self.monitor == MONITOR_TRAIN_HOLDOUT:
            self._eval_split_consumed = False
            if self._live_metrics is not None:
                self._live_metrics["on_eval_end_ignored"] = True
                self._live_metrics["eval_split_read_for_selection"] = False
                audit = self._live_metrics.get("holdout_audit")
                if isinstance(audit, dict):
                    audit["on_eval_end_ignored"] = True
                    audit["eval_split_read_for_selection"] = False
            return
        self._eval_split_consumed = True
        if self._live_metrics is not None:
            self._live_metrics["on_eval_end_ignored"] = False
            self._live_metrics["eval_split_read_for_selection"] = True
        extra = eval_metrics.get("extra", {}) if isinstance(eval_metrics.get("extra", {}), dict) else {}
        rows = extra.get("forgetting_by_segment") or []
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            sid = int(row.get("segment_id", -1))
            if sid < 0:
                continue
            self._forgetting_by_segment[sid] = float(row.get("forgetting", 0.0))

    def _training_examples(self, *, segment: Segment, model: Any, lora: Any) -> List[Example]:
        if self.monitor == MONITOR_TRAIN_HOLDOUT:
            fit, hold, hold_idx = split_train_holdout(
                segment.train,
                segment_id=int(segment.segment_id),
                seed=int(self.holdout_seed),
                holdout_n=int(self.holdout_n),
            )
            self._current_fit = fit
            self._current_holdout = hold
            self._holdouts[int(segment.segment_id)] = list(hold)
            self._current_memory = self._build_memory(segment, model, lora, source_examples=fit)
            mixed = self._mix_current_and_replay(segment, current=fit)
            self._last_audit = self._holdout_audit(segment, fit, hold, hold_idx, mixed)
            return mixed
        self._current_fit = list(segment.train)
        self._current_holdout = []
        self._current_memory = self._build_memory(segment, model, lora)
        mixed = self._mix_current_and_replay(segment)
        self._last_audit = {
            "segment_id": int(segment.segment_id),
            "selector_version": MONITOR_EVAL_SPLIT,
            "eval_split_read_for_selection": True,
        }
        return mixed

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
        self._n_replay_used = 0
        self._n_neighbor_used = 0
        self._n_forgotten_used = 0
        self._n_anti_used = 0
        self._n_uniform_used = 0
        self._n_related_used = 0
        self._n_degraded_used = 0
        self._fallback_count = 0
        self._n_teacher_proto = 0
        self._last_replay_log = {}
        metrics = super().train_on_segment(
            segment=segment, model=model, lora=lora, lr=lr, epochs=epochs, batch_size=batch_size
        )
        if self._current_memory is not None:
            mem = self._current_memory
        elif self.monitor == MONITOR_TRAIN_HOLDOUT:
            mem = self._build_memory(segment, model, lora, source_examples=self._current_fit)
        else:
            mem = self._build_memory(segment, model, lora)
        self._memories.append(mem)
        self._current_memory = None
        if self.monitor == MONITOR_TRAIN_HOLDOUT:
            self._refresh_holdout_forgetting(model)
        self._live_metrics = metrics
        metrics["replay_strategy"] = self.strategy
        metrics["transfer_slot"] = self.transfer_slot
        metrics["replay_ratio"] = float(self.replay_ratio)
        metrics["neighbor_ratio"] = float(self.neighbor_ratio)
        metrics["forgotten_ratio"] = float(self.forgotten_ratio)
        metrics["coverage_dedup"] = bool(self.coverage_dedup)
        metrics["buffer_k"] = int(self.buffer_k)
        metrics["teacher_proto_mode"] = self.teacher_proto_mode
        metrics["monitor"] = self.monitor
        metrics["selector_version"] = self.monitor
        metrics["holdout_n"] = int(self.holdout_n) if self.monitor == MONITOR_TRAIN_HOLDOUT else 0
        metrics["holdout_seed"] = int(self.holdout_seed) if self.monitor == MONITOR_TRAIN_HOLDOUT else 0
        if self._last_audit:
            metrics["holdout_audit"] = self._last_audit
            metrics["holdout_in_train"] = bool(self._last_audit.get("holdout_in_train", False))
            metrics["holdout_in_memory"] = bool(self._last_audit.get("holdout_in_memory", False))
            metrics["eval_split_read_for_selection"] = bool(
                self._last_audit.get("eval_split_read_for_selection", self.monitor == MONITOR_EVAL_SPLIT)
            )
        metrics["n_replay"] = int(self._n_replay_used)
        metrics["n_neighbor_replay"] = int(self._n_neighbor_used)
        metrics["n_forgotten_replay"] = int(self._n_forgotten_used)
        metrics["n_anti_replay"] = int(self._n_anti_used)
        metrics["n_uniform_replay"] = int(self._n_uniform_used)
        metrics["n_related_replay"] = int(self._n_related_used)
        metrics["n_degraded_replay"] = int(self._n_degraded_used)
        metrics["fallback_count"] = int(self._fallback_count)
        metrics["budget_policy"] = self.budget_policy
        metrics["ratio_definition"] = self.ratio_definition
        metrics["target_fraction"] = float(self.target_fraction)
        metrics["n_teacher_proto"] = int(self._n_teacher_proto)
        metrics["memory_segments"] = len(self._memories)
        metrics["train.n_steps"] = int(metrics.get("batches", 0))
        log = dict(self._last_replay_log)
        if log:
            steps = int(metrics.get("batches", 0))
            log["train_steps"] = steps
            if not isinstance(metrics.get("num_total_tokens"), (int, float)):
                log["num_total_tokens"] = None
                log["token_count_note"] = (
                    "train_steps is the optimizer step count for this segment. "
                    "ce_opsd_replay does not accumulate num_total_tokens; the missing total is not estimated."
                )
            self._last_replay_log = log
            metrics["replay_log"] = log
            metrics["replay_unique_pairs"] = int(log.get("unique_pairs", 0))
            metrics["replay_unique_instructions"] = int(log.get("unique_instructions", 0))
            metrics["replay_top1_instruction_share"] = float(log.get("top1_instruction_share", 0.0))
            metrics["replay_neighbor_top1_instruction_share"] = float(
                log.get("neighbor_top1_instruction_share", 0.0)
            )
        return metrics

    def dump_segment_state(self) -> Dict[str, Any]:
        """Adapter-side dump of memories + last eval forgetting (for transfer isolation)."""
        return {
            "strategy": self.strategy,
            "transfer_slot": self.transfer_slot,
            "coverage_dedup": bool(self.coverage_dedup),
            "buffer_k": int(self.buffer_k),
            "forgetting_by_segment": {str(k): float(v) for k, v in self._forgetting_by_segment.items()},
            "memories": [self._memory_to_dict(mem) for mem in self._memories],
        }

    def load_memory_dump(self, path: str) -> None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"replay.init_memory_path not found: {p}")
        payload = json.loads(p.read_text(encoding="utf-8"))
        forgetting = payload.get("forgetting_by_segment") or {}
        self._forgetting_by_segment = {int(k): float(v) for k, v in forgetting.items()}
        self._memories = [self._memory_from_dict(row) for row in (payload.get("memories") or [])]

    def _teacher_prototypes_for(self, ex: Example) -> Optional[List[Dict[str, Any]]]:
        if self.teacher_proto_mode == "none" or not self._memories:
            return None
        n = random.choice([1, 2])
        if self.teacher_proto_mode == "random":
            pool = [item for mem in self._memories for item in mem.examples]
        else:
            pool = self._compatible_examples(ex)
            if not pool:
                pool = [item for mem in self._memories for item in mem.examples]
        if not pool:
            return None
        picked = random.sample(pool, k=min(n, len(pool)))
        self._n_teacher_proto += len(picked)
        return [self._example_to_proto(item, mem) for item, mem in ((p, self._memory_of(p)) for p in picked)]

    def _mix_current_and_replay(self, segment: Segment, current: Optional[List[Example]] = None) -> List[Example]:
        current = list(segment.train if current is None else current)
        picks: List[ReplayPick] = []
        if self.strategy in _UNIFIED_STRATEGIES:
            if not self._memories:
                self._last_replay_log = self._empty_replay_log(segment, n_current=len(current))
                return current
            return self._mix_unified(segment, current)
        if not self._memories or self.replay_ratio <= 0:
            self._last_replay_log = self._coverage_stats(segment, picks, n_current=len(current))
            self._attach_budget_fields(
                n_current=len(current),
                n_replay_target=0,
                picks=picks,
                ranked_segment_ids=[],
                selector="legacy_inactive",
                fallback_count=0,
                allocator_related=0,
                allocator_degraded=0,
            )
            return current
        n = len(current)
        if self.strategy == "uniform":
            n_replay = quota(n, self.replay_ratio)
            pool = [ex for mem in self._memories for ex in mem.examples]
            replay = _sample_examples(pool, n_replay)
            picks = self._tag_picks(replay, "uniform")
            self._n_replay_used = len(replay)
            self._n_uniform_used = len(replay)
            n_target = int(n_replay)
            selector = "legacy_uniform"
            alloc_related, alloc_degraded = 0, 0
        else:
            n_nb = quota(n, self.neighbor_ratio)
            n_fg = quota(n, self.forgotten_ratio)
            transfer = self._select_transfer_slot(n_nb)
            forgotten = self._select_forgotten(n_fg)
            slot_name = self.transfer_slot if self.transfer_slot in {"anti", "uniform"} else "neighbor"
            picks = self._tag_picks(transfer, slot_name) + self._tag_picks(forgotten, "forgotten")
            replay = transfer + forgotten
            if slot_name == "neighbor":
                self._n_neighbor_used = len(transfer)
            elif slot_name == "anti":
                self._n_anti_used = len(transfer)
            else:
                self._n_uniform_used = len(transfer)
            self._n_forgotten_used = len(forgotten)
            self._n_replay_used = len(replay)
            n_target = int(n_nb + n_fg)
            selector = "legacy_proto"
            alloc_related, alloc_degraded = int(n_nb), int(n_fg)
        self._last_replay_log = self._coverage_stats(segment, picks, n_current=len(current))
        self._attach_budget_fields(
            n_current=len(current),
            n_replay_target=n_target,
            picks=picks,
            ranked_segment_ids=self._legacy_ranked_ids(),
            selector=selector,
            fallback_count=0,
            allocator_related=alloc_related,
            allocator_degraded=alloc_degraded,
        )
        mixed = current + replay
        random.shuffle(mixed)
        return mixed

    def _mix_unified(self, segment: Segment, current: List[Example]) -> List[Example]:
        """Random, compat_uniform and true_prototype share one mixed-batch count."""
        budget = allocate_mixed_batch(len(current), rho=float(self.target_fraction))
        if budget.n_related + budget.n_degraded != budget.n_replay_target:
            raise RuntimeError(
                f"slot split {budget.n_related}+{budget.n_degraded} != {budget.n_replay_target}"
            )
        fallback_count = 0
        ranked: List[int] = []
        if self.strategy == "random":
            pool = [ex for mem in self._memories for ex in mem.examples]
            replay = _sample_examples(pool, budget.n_replay_target)
            picks = self._tag_picks(replay, "random")
            self._n_uniform_used = len(replay)
            selector = "uniform_all_history"
            ranked = sorted(int(mem.segment_id) for mem in self._memories)
            n_related_used = 0
            n_degraded_used = 0
        elif self.strategy == "compat_uniform":
            related, related_fallback = self._select_compat_uniform_related(budget.n_related)
            degraded = self._select_degraded_excluding(budget.n_degraded, {id(ex) for ex in related})
            fallback_count = int(related_fallback)
            picks = self._tag_picks(related, "related") + self._tag_picks(degraded, "degraded")
            selector = "format_compatible_uniform"
            ranked = self._compatible_ids_by_segment()
            n_related_used = len(related)
            n_degraded_used = len(degraded)
            self._n_neighbor_used = n_related_used
            self._n_forgotten_used = n_degraded_used
        elif self.strategy == "true_prototype":
            related, related_fallback, ranked = self._select_true_prototype_related(budget.n_related)
            degraded = self._select_degraded_excluding(budget.n_degraded, {id(ex) for ex in related})
            fallback_count = int(related_fallback)
            picks = self._tag_picks(related, "related") + self._tag_picks(degraded, "degraded")
            selector = "l2_cosine_descending"
            n_related_used = len(related)
            n_degraded_used = len(degraded)
            self._n_neighbor_used = n_related_used
            self._n_forgotten_used = n_degraded_used
        else:
            raise ValueError(f"not a unified strategy: {self.strategy}")
        replay = [p.example for p in picks]
        self._n_replay_used = len(replay)
        self._n_related_used = int(n_related_used)
        self._n_degraded_used = int(n_degraded_used)
        self._fallback_count = int(fallback_count)
        self._last_replay_log = self._coverage_stats(segment, picks, n_current=len(current))
        self._attach_budget_fields(
            n_current=len(current),
            n_replay_target=int(budget.n_replay_target),
            picks=picks,
            ranked_segment_ids=ranked,
            selector=selector,
            fallback_count=int(fallback_count),
            allocator_related=int(budget.n_related),
            allocator_degraded=int(budget.n_degraded),
            protocol_fraction=float(budget.realized_replay_fraction),
        )
        mixed = current + replay
        random.shuffle(mixed)
        return mixed

    def _select_compat_uniform_related(self, n: int) -> tuple:
        """Format-compatible uniform draw. Cosine is not consulted."""
        if n <= 0 or not self._memories:
            return [], 0
        pool: List[Example] = []
        if self._current_memory is not None:
            for mem in self._memories:
                if _format_compatible(self._current_memory, mem):
                    pool.extend(mem.examples)
        fallback = 0
        if not pool:
            pool = [ex for mem in self._memories for ex in mem.examples]
            picked = _sample_examples(pool, n)
            return picked, len(picked)
        picked = _sample_examples(pool, n)
        if len(picked) < n:
            used = {id(ex) for ex in picked}
            rest = [ex for mem in self._memories for ex in mem.examples if id(ex) not in used]
            extra = _sample_examples(rest, n - len(picked))
            fallback = len(extra)
            picked.extend(extra)
        return picked, fallback

    def _select_true_prototype_related(self, n: int) -> tuple:
        """Fill Related from explicit L2 cosine, then sample inside the segment buffer.

        Cached frozen instruction vectors are reused. This method does not call
        the encoder, does not read gold text for the rank, and does not read
        ``segment.eval``. Ties break by ascending ``segment_id``.
        """
        if n <= 0 or not self._memories:
            return [], 0, []
        ranked_scores: List[tuple] = []
        if self._current_memory is not None:
            for mem in self._memories:
                if not _format_compatible(self._current_memory, mem):
                    continue
                score = cosine_similarity(self._current_memory.embedding, mem.embedding)
                ranked_scores.append((int(mem.segment_id), float(score)))
        order = rank_segment_ids(ranked_scores)
        by_id = {int(mem.segment_id): mem for mem in self._memories}
        picked: List[Example] = []
        used: set = set()
        for sid in order:
            if len(picked) >= n:
                break
            mem = by_id[sid]
            avail = [ex for ex in mem.examples if id(ex) not in used]
            chunk = _sample_examples(avail, n - len(picked))
            for ex in chunk:
                used.add(id(ex))
            picked.extend(chunk)
        fallback = 0
        if len(picked) < n:
            rest = [ex for mem in self._memories for ex in mem.examples if id(ex) not in used]
            extra = _sample_examples(rest, n - len(picked))
            fallback = len(extra)
            picked.extend(extra)
        return picked, fallback, order

    def _select_degraded_excluding(self, n: int, banned_ids: set) -> List[Example]:
        """Train-holdout forgetting slot, without repeating Related rows.

        The forgotten segment is chosen the same way as ``select_forgotten``.
        Rows already used by Related are skipped. If that segment cannot fill
        the integer quota, the remainder comes from other historical memory.
        That refill does not re-round the budget.
        """
        if n <= 0 or not self._memories:
            return []
        banned = set(banned_ids)
        best_sid = None
        best_f = -1.0
        known = {int(mem.segment_id) for mem in self._memories}
        for sid, fgt in self._forgetting_by_segment.items():
            if int(sid) in known and float(fgt) > best_f:
                best_f = float(fgt)
                best_sid = int(sid)
        if best_sid is None or best_f <= 0:
            pool = [ex for mem in self._memories for ex in mem.examples if id(ex) not in banned]
            return _sample_examples(pool, n)
        primary: List[Example] = []
        for mem in self._memories:
            if int(mem.segment_id) == best_sid:
                primary = [ex for ex in mem.examples if id(ex) not in banned]
                break
        picked = _sample_examples(primary, n)
        if len(picked) >= n:
            return picked
        used = set(banned)
        used.update(id(ex) for ex in picked)
        rest = [ex for mem in self._memories for ex in mem.examples if id(ex) not in used]
        picked.extend(_sample_examples(rest, n - len(picked)))
        return picked

    def _compatible_ids_by_segment(self) -> List[int]:
        """Segment ids only. compat_uniform must not publish a cosine ranking."""
        if self._current_memory is None:
            return sorted(int(mem.segment_id) for mem in self._memories)
        ids = [
            int(mem.segment_id)
            for mem in self._memories
            if _format_compatible(self._current_memory, mem)
        ]
        return sorted(ids)

    def _legacy_ranked_ids(self) -> List[int]:
        """Diagnostic order for the old proto path. Selection itself stays uniform."""
        if self.strategy != "proto" or self._current_memory is None:
            return sorted(int(mem.segment_id) for mem in self._memories)
        scored: List[tuple] = []
        for mem in self._memories:
            if not _format_compatible(self._current_memory, mem):
                continue
            scored.append((int(mem.segment_id), _cosine(self._current_memory.embedding, mem.embedding)))
        return rank_segment_ids(scored)

    def _candidate_cosine_rows(self) -> List[Dict[str, Any]]:
        if self._current_memory is None:
            return []
        rows: List[Dict[str, Any]] = []
        for mem in self._memories:
            rows.append(
                {
                    "segment_id": int(mem.segment_id),
                    "cosine": float(cosine_similarity(self._current_memory.embedding, mem.embedding)),
                    "raw_dot": float(_cosine(self._current_memory.embedding, mem.embedding)),
                    "embedding_l2": float(vector_l2(mem.embedding)),
                    "format_compatible": bool(_format_compatible(self._current_memory, mem)),
                }
            )
        rows.sort(key=lambda row: int(row["segment_id"]))
        return rows

    def _selected_similarity(self, picks: List[ReplayPick]) -> tuple:
        if self._current_memory is None or not picks:
            return None, None, None, []
        by_id = {int(mem.segment_id): mem for mem in self._memories}
        values: List[float] = []
        for pick in picks:
            mem = by_id.get(int(pick.source_segment_id))
            if mem is None or not mem.embedding or not self._current_memory.embedding:
                continue
            values.append(float(cosine_similarity(self._current_memory.embedding, mem.embedding)))
        if not values:
            return None, None, None, []
        return (
            float(sum(values) / len(values)),
            float(min(values)),
            float(max(values)),
            values,
        )

    def _empty_replay_log(self, segment: Segment, *, n_current: int) -> Dict[str, Any]:
        self._last_replay_log = self._coverage_stats(segment, [], n_current=n_current)
        self._attach_budget_fields(
            n_current=n_current,
            n_replay_target=0,
            picks=[],
            ranked_segment_ids=[],
            selector="no_history",
            fallback_count=0,
            allocator_related=0,
            allocator_degraded=0,
        )
        return dict(self._last_replay_log)

    def _attach_budget_fields(
        self,
        *,
        n_current: int,
        n_replay_target: int,
        picks: List[ReplayPick],
        ranked_segment_ids: List[int],
        selector: str,
        fallback_count: int,
        allocator_related: int,
        allocator_degraded: int,
        protocol_fraction: Optional[float] = None,
    ) -> None:
        actual = len(picks)
        pair_keys = {(p.instruction, p.input) for p in picks}
        mean_s, min_s, max_s, selected_scores = self._selected_similarity(picks)
        norms: List[float] = []
        if self._current_memory is not None and self._current_memory.embedding:
            norms.append(vector_l2(self._current_memory.embedding))
        for mem in self._memories:
            if mem.embedding:
                norms.append(vector_l2(mem.embedding))
        mean_l2 = float(sum(norms) / len(norms)) if norms else None
        appear_unit = bool(norms) and all(abs(n - 1.0) <= 1e-3 for n in norms)
        fraction = realized_replay_fraction(actual, n_current)
        related_ids = [int(p.source_segment_id) for p in picks if p.pool == "related"]
        note = {
            "random": "uniform draw over all historical memory; similarity is logged and not used",
            "uniform_all_history": "uniform draw over all historical memory; similarity is logged and not used",
            "compat_uniform": "format-compatible uniform Related plus holdout-forgetting Degraded; not a cosine prototype",
            "format_compatible_uniform": "format-compatible uniform Related plus holdout-forgetting Degraded; not a cosine prototype",
            "true_prototype": "Related slots filled by explicit L2 cosine of cached frozen instruction vectors",
            "l2_cosine_descending": "Related slots filled by explicit L2 cosine of cached frozen instruction vectors",
            "legacy_uniform": "legacy quota(n, replay_ratio); not the mixed-batch 11/56 budget",
            "legacy_proto": "legacy separate round of neighbor_ratio and forgotten_ratio; not true prototype",
            "legacy_inactive": "no replay drawn",
            "no_history": "no historical memory yet",
        }.get(selector, selector)
        fields = {
            "n_replay_target": int(n_replay_target),
            "n_replay_actual": int(actual),
            "n_related": int(sum(1 for p in picks if p.pool == "related")),
            "n_degraded": int(sum(1 for p in picks if p.pool == "degraded")),
            "n_unique": int(len(pair_keys)),
            "realized_replay_fraction": float(fraction),
            "target_replay_fraction": (
                float(protocol_fraction)
                if protocol_fraction is not None
                else float(realized_replay_fraction(n_replay_target, n_current))
            ),
            "shortfall": int(max(0, int(n_replay_target) - actual)),
            "ranked_segment_ids": [int(x) for x in ranked_segment_ids],
            "candidate_cosine_scores": self._candidate_cosine_rows(),
            "selected_similarity_mean": mean_s,
            "selected_similarity_min": min_s,
            "selected_similarity_max": max_s,
            "selected_cosine_scores": selected_scores,
            "related_source_segment_ids": related_ids,
            "fallback_count": int(fallback_count),
            "allocator_n_related": int(allocator_related),
            "allocator_n_degraded": int(allocator_degraded),
            "budget_policy": self.budget_policy,
            "ratio_definition": self.ratio_definition if self.strategy in _UNIFIED_STRATEGIES else self.ratio_definition,
            "selector": selector,
            "selector_note": note,
            "selector_l2_renormalizes": bool(self.strategy == "true_prototype"),
            "cached_embedding_l2_mean": mean_l2,
            "cached_embeddings_appear_l2_normalized": appear_unit,
        }
        if self._last_replay_log is None:
            self._last_replay_log = {}
        self._last_replay_log.update(fields)

    def _select_transfer_slot(self, n: int) -> List[Example]:
        if self.transfer_slot == "uniform":
            if n <= 0 or not self._memories:
                return []
            pool = [ex for mem in self._memories for ex in mem.examples]
            return self._sample_transfer(pool, n)
        if self.transfer_slot == "anti":
            return self._select_anti_neighbors(n)
        return self._select_neighbors(n)

    def _sample_transfer(self, pool: List[Example], n: int) -> List[Example]:
        """Neighbor/transfer draw. Forgotten pool never uses this."""
        if self.coverage_dedup:
            return self._sample_coverage_dedup(pool, n)
        return _sample_examples(pool, n)

    def _sample_coverage_dedup(self, pool: List[Example], n: int) -> List[Example]:
        """In-batch unique (instruction, input); prefer low cover count then LRU.

        Task set is already chosen by the caller. Quota ``n`` is unchanged; if
        unique pairs are fewer than ``n``, remaining slots refill from the same
        pool (still less-covered first).
        """
        if n <= 0 or not pool:
            return []
        by_pair: Dict[tuple, List[Example]] = {}
        for ex in pool:
            by_pair.setdefault(_pair_key(ex), []).append(ex)
        keys = list(by_pair.keys())
        random.shuffle(keys)
        keys.sort(
            key=lambda k: (
                int(self._pair_cover_n.get(k, 0)),
                int(self._pair_last_tick.get(k, -1)),
            )
        )
        picked: List[Example] = []
        for key in keys:
            if len(picked) >= n:
                break
            picked.append(random.choice(by_pair[key]))
        if len(picked) < n:
            seen = {id(ex) for ex in picked}
            remain = [ex for ex in pool if id(ex) not in seen]
            random.shuffle(remain)
            remain.sort(
                key=lambda ex: (
                    int(self._pair_cover_n.get(_pair_key(ex), 0)),
                    int(self._pair_last_tick.get(_pair_key(ex), -1)),
                )
            )
            for ex in remain:
                if len(picked) >= n:
                    break
                picked.append(ex)
        for ex in picked:
            key = _pair_key(ex)
            self._pair_cover_n[key] = int(self._pair_cover_n.get(key, 0)) + 1
            self._cover_tick += 1
            self._pair_last_tick[key] = int(self._cover_tick)
        return picked

    def _select_neighbors(self, n: int) -> List[Example]:
        """v2 proto neighbor: format-compatible then uniform (cosine only orders the pool)."""
        if n <= 0 or self._current_memory is None:
            return []
        ranked: List[tuple] = []
        cur = self._current_memory
        for mem in self._memories:
            if not _format_compatible(cur, mem):
                continue
            ranked.append((_cosine(cur.embedding, mem.embedding), mem))
        if not ranked:
            pool = [ex for mem in self._memories for ex in mem.examples]
            return self._sample_transfer(pool, n)
        ranked.sort(key=lambda x: x[0], reverse=True)
        pool: List[Example] = []
        for _, mem in ranked:
            pool.extend(mem.examples)
        return self._sample_transfer(pool, n)

    def _select_anti_neighbors(self, n: int) -> List[Example]:
        """Format-compatible segment with lowest cosine; sample its K-buffer."""
        if n <= 0 or self._current_memory is None:
            return []
        ranked: List[tuple] = []
        cur = self._current_memory
        for mem in self._memories:
            if not _format_compatible(cur, mem):
                continue
            ranked.append((_cosine(cur.embedding, mem.embedding), mem))
        if not ranked:
            pool = [ex for mem in self._memories for ex in mem.examples]
            return self._sample_transfer(pool, n)
        ranked.sort(key=lambda x: x[0])
        return self._sample_transfer(ranked[0][1].examples, n)

    def _select_forgotten(self, n: int) -> List[Example]:
        # Version is self.monitor: eval_split_monitor reads on_eval_end scores;
        # train_holdout reads only holdout scores written by _refresh_holdout_forgetting.
        return select_forgotten(
            self._memories,
            self._forgetting_by_segment,
            n,
            selector_version=self.monitor,
            sample_fn=_sample_examples,
        )

    def _refresh_holdout_forgetting(self, model: Any) -> None:
        """Score stored train holdouts and write forgetting. Never receives segment.eval."""
        if self.monitor != MONITOR_TRAIN_HOLDOUT or not self._holdouts:
            return
        norm = self.cfg.get("eval_normalization") if isinstance(self.cfg.get("eval_normalization"), dict) else {}
        model_cfg = self.cfg.get("model") if isinstance(self.cfg.get("model"), dict) else {}
        max_new = int(model_cfg.get("gen_max_new_tokens", 64))
        per_seg = score_holdout_map(
            model,
            self._holdouts,
            normalization_cfg=norm,
            max_new_tokens=max_new,
        )
        rows, mean_f, best = forgetting_from_accuracies(per_seg, self._holdout_best)
        self._holdout_best = best
        self._forgetting_by_segment = {int(row["segment_id"]): float(row["forgetting"]) for row in rows}
        self._last_audit["holdout_forgetting_by_segment"] = rows
        self._last_audit["holdout_forgetting_mean"] = float(mean_f)
        self._last_audit["eval_split_read_for_selection"] = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _holdout_audit(
        self,
        segment: Segment,
        fit: List[Example],
        hold: List[Example],
        hold_idx: List[int],
        mixed: List[Example],
    ) -> Dict[str, Any]:
        hold_keys = [list(example_key(ex)) for ex in hold]
        all_hold = [ex for rows in self._holdouts.values() for ex in rows]
        hold_ids = {id(ex) for ex in all_hold}
        memories = list(self._memories)
        if self._current_memory is not None:
            memories.append(self._current_memory)
        mem_examples = [ex for mem in memories for ex in mem.examples]
        in_train = any(id(ex) in hold_ids for ex in mixed)
        in_memory = any(id(ex) in hold_ids for ex in mem_examples)
        return {
            "segment_id": int(segment.segment_id),
            "selector_version": MONITOR_TRAIN_HOLDOUT,
            "holdout_n": len(hold),
            "fit_n": len(fit),
            "holdout_indices": [int(i) for i in hold_idx],
            "holdout_keys": hold_keys,
            "n_train_examples": len(mixed),
            "holdout_in_train": bool(in_train),
            "holdout_in_memory": bool(in_memory),
            "eval_split_read_for_selection": False,
            "official_eval_n": len(segment.eval),
        }

    def _compatible_examples(self, ex: Example) -> List[Example]:
        probe = _example_format(ex)
        out: List[Example] = []
        for mem in self._memories:
            if mem.task_score_type != probe["task_score_type"]:
                continue
            if not _length_compatible(probe["mean_gold_len"], mem.mean_gold_len):
                continue
            out.extend(mem.examples)
        return out

    def _build_memory(
        self,
        segment: Segment,
        model: Any,
        lora: Any,
        source_examples: Optional[List[Example]] = None,
    ) -> SegmentMemory:
        pool = list(segment.train if source_examples is None else source_examples)
        instruction = str(pool[0].instruction) if pool else str(segment.segment_name)
        score_types: List[str] = []
        gold_lens: List[float] = []
        for ex in pool:
            fmt = _example_format(ex)
            score_types.append(fmt["task_score_type"])
            gold_lens.append(fmt["mean_gold_len"])
        task_score_type = _majority(score_types) if score_types else "strict_em"
        mean_gold_len = sum(gold_lens) / max(1, len(gold_lens))
        embedding = _frozen_instruction_embed(model, lora, instruction)
        examples = _pick_diverse_k(pool, self.buffer_k)
        return SegmentMemory(
            segment_id=int(segment.segment_id),
            instruction=instruction,
            task_score_type=task_score_type,
            mean_gold_len=float(mean_gold_len),
            embedding=embedding,
            examples=examples,
        )

    def _memory_of(self, ex: Example) -> Optional[SegmentMemory]:
        for mem in self._memories:
            if ex in mem.examples:
                return mem
        return None

    def _tag_picks(self, examples: List[Example], pool: str) -> List[ReplayPick]:
        out: List[ReplayPick] = []
        for ex in examples:
            mem = self._memory_of(ex)
            sid = int(mem.segment_id) if mem is not None else -1
            out.append(
                ReplayPick(
                    example=ex,
                    pool=pool,
                    source_segment_id=sid,
                    instruction=str(ex.instruction),
                    input=str(ex.input),
                )
            )
        return out

    def _coverage_stats(
        self, segment: Segment, picks: List[ReplayPick], *, n_current: int
    ) -> Dict[str, Any]:
        pair_keys = [(p.instruction, p.input) for p in picks]
        unique_pairs = len(set(pair_keys))
        instr_counts = Counter(p.instruction for p in picks)
        unique_instructions = len(instr_counts)
        top1_instr, top1_n = ("", 0)
        if instr_counts:
            top1_instr, top1_n = instr_counts.most_common(1)[0]
        neighbor_instr = Counter(p.instruction for p in picks if p.pool == "neighbor")
        nb_top1_share = 0.0
        if neighbor_instr:
            nb_top1_share = float(neighbor_instr.most_common(1)[0][1]) / max(1, sum(neighbor_instr.values()))
        per_seg: Dict[int, Dict[str, int]] = {}
        for p in picks:
            row = per_seg.setdefault(
                int(p.source_segment_id),
                {
                    "segment_id": int(p.source_segment_id),
                    "n_replay": 0,
                    "n_neighbor": 0,
                    "n_forgotten": 0,
                    "n_anti": 0,
                    "n_uniform": 0,
                    "n_related": 0,
                    "n_degraded": 0,
                    "n_random": 0,
                },
            )
            row["n_replay"] += 1
            if p.pool in {"neighbor", "forgotten", "anti", "uniform", "related", "degraded", "random"}:
                row[f"n_{p.pool}"] += 1
        n_replay = len(picks)
        return {
            "segment_id": int(segment.segment_id),
            "segment_name": str(segment.segment_name),
            "strategy": self.strategy,
            "transfer_slot": self.transfer_slot,
            "coverage_dedup": bool(self.coverage_dedup),
            "n_current": int(n_current),
            "n_replay": int(n_replay),
            "n_neighbor": int(sum(1 for p in picks if p.pool == "neighbor")),
            "n_forgotten": int(sum(1 for p in picks if p.pool == "forgotten")),
            "n_anti": int(sum(1 for p in picks if p.pool == "anti")),
            "n_uniform": int(sum(1 for p in picks if p.pool == "uniform")),
            "unique_pairs": int(unique_pairs),
            "unique_instructions": int(unique_instructions),
            "top1_instruction": top1_instr[:240],
            "top1_instruction_share": float(top1_n) / max(1, n_replay),
            "neighbor_top1_instruction_share": float(nb_top1_share),
            "per_segment": [per_seg[k] for k in sorted(per_seg)],
            "lambda_opsd": float(self.lambda_opsd),
            "opsd_skipped_lambda": bool(float(self.lambda_opsd) <= 0.0),
        }

    def _example_to_proto(self, ex: Example, mem: Optional[SegmentMemory]) -> Dict[str, Any]:
        golds = _example_golds(ex)
        return {
            "instruction": ex.instruction,
            "input": ex.input,
            "output": golds[0] if golds else "",
            "task_score_type": mem.task_score_type if mem else _example_format(ex)["task_score_type"],
            "mean_gold_len": mem.mean_gold_len if mem else _example_format(ex)["mean_gold_len"],
        }

    def _memory_to_dict(self, mem: SegmentMemory) -> Dict[str, Any]:
        return {
            "segment_id": int(mem.segment_id),
            "instruction": mem.instruction,
            "task_score_type": mem.task_score_type,
            "mean_gold_len": float(mem.mean_gold_len),
            "embedding": [float(x) for x in mem.embedding],
            "examples": [_example_to_dict(ex) for ex in mem.examples],
        }

    def _memory_from_dict(self, row: Dict[str, Any]) -> SegmentMemory:
        examples = [_example_from_dict(ex) for ex in (row.get("examples") or [])]
        return SegmentMemory(
            segment_id=int(row.get("segment_id", -1)),
            instruction=str(row.get("instruction", "")),
            task_score_type=str(row.get("task_score_type", "strict_em")),
            mean_gold_len=float(row.get("mean_gold_len", 0.0)),
            embedding=[float(x) for x in (row.get("embedding") or [])],
            examples=examples,
        )


def _pair_key(ex: Example) -> tuple:
    return (str(ex.instruction), str(ex.input))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return bool(default)


def _example_to_dict(ex: Example) -> Dict[str, Any]:
    return {
        "instruction": ex.instruction,
        "input": ex.input,
        "output": ex.output,
        "outputs": list(ex.outputs or []),
    }


def _example_from_dict(row: Dict[str, Any]) -> Example:
    return Example(
        instruction=str(row.get("instruction", "")),
        input=str(row.get("input", "")),
        output=str(row.get("output", "")),
        outputs=[str(x) for x in (row.get("outputs") or [])],
    )


def _example_format(ex: Example) -> Dict[str, Any]:
    golds = _example_golds(ex)
    gold = golds[0] if golds else ""
    info = _score_task_aware(
        pred=gold,
        gold=gold,
        norm_pred=gold,
        norm_gold=gold,
        instruction=ex.instruction,
        input_text=ex.input,
        cfg={},
    )
    mean_len = sum(len(str(g).split()) for g in golds) / max(1, len(golds))
    return {
        "task_score_type": str(info.get("task_score_type", "strict_em")),
        "mean_gold_len": float(mean_len),
    }


def _format_compatible(a: SegmentMemory, b: SegmentMemory) -> bool:
    if a.task_score_type != b.task_score_type:
        return False
    return _length_compatible(a.mean_gold_len, b.mean_gold_len)


def _length_compatible(a: float, b: float) -> bool:
    lo, hi = (float(a), float(b)) if a <= b else (float(b), float(a))
    if lo <= 0:
        return hi <= 0 or hi <= 2.0
    return (hi / lo) <= 2.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    return float(sum(float(a[i]) * float(b[i]) for i in range(n)))


def _majority(items: List[str]) -> str:
    counts: Dict[str, int] = {}
    for x in items:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get) if counts else "strict_em"


def _sample_examples(pool: List[Example], n: int) -> List[Example]:
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        return list(pool)
    return random.sample(pool, k=n)


def _pick_diverse_k(examples: List[Example], k: int) -> List[Example]:
    if k <= 0 or not examples:
        return []
    by_input: Dict[str, List[Example]] = {}
    for ex in examples:
        by_input.setdefault(str(ex.input), []).append(ex)
    keys = list(by_input.keys())
    random.shuffle(keys)
    picked: List[Example] = []
    idx = 0
    while len(picked) < min(k, len(examples)) and keys:
        key = keys[idx % len(keys)]
        bucket = by_input[key]
        if bucket:
            picked.append(bucket.pop())
        idx += 1
        if idx > len(keys) * max(1, k) + 8:
            break
    if len(picked) < min(k, len(examples)):
        remain = [ex for ex in examples if ex not in picked]
        picked.extend(remain[: min(k, len(examples)) - len(picked)])
    return picked


def _frozen_instruction_embed(model: Any, lora: Any, instruction: str) -> List[float]:
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is not None and hasattr(peft_model, "disable_adapter"):
        with peft_model.disable_adapter():
            return list(model.get_activations([instruction])[0])
    return list(model.get_activations([instruction])[0])
