from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from baselines.basic_baselines.ce_opsd.method import CeOpsdMethod, _example_golds
from core.data import Example, Segment
from core.evaluate import _score_task_aware


@dataclass
class SegmentMemory:
    segment_id: int
    instruction: str
    task_score_type: str
    mean_gold_len: float
    embedding: List[float]
    examples: List[Example] = field(default_factory=list)


class CeOpsdReplayMethod(CeOpsdMethod):
    """
    Same-step CE+OPSD with 20% replay and optional teacher prototype injection.

    replay.strategy:
      - uniform: 20% random from per-segment K=10 buffers
      - proto: 10% format-compatible nearest + 10% most-forgotten segment
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
        if self.strategy not in {"uniform", "proto"}:
            raise ValueError(f"replay.strategy must be 'uniform' or 'proto', got: {self.strategy}")
        self.neighbor_ratio = float(replay_cfg.get("neighbor_ratio", 0.1))
        self.forgotten_ratio = float(replay_cfg.get("forgotten_ratio", 0.1))
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
        self._n_teacher_proto = 0

    def on_eval_end(self, eval_metrics: Dict[str, Any]) -> None:
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
        self._current_memory = self._build_memory(segment, model, lora)
        mixed = self._mix_current_and_replay(segment)
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
        self._n_teacher_proto = 0
        metrics = super().train_on_segment(
            segment=segment, model=model, lora=lora, lr=lr, epochs=epochs, batch_size=batch_size
        )
        mem = self._current_memory or self._build_memory(segment, model, lora)
        self._memories.append(mem)
        self._current_memory = None
        metrics["replay_strategy"] = self.strategy
        metrics["replay_ratio"] = float(self.replay_ratio)
        metrics["buffer_k"] = int(self.buffer_k)
        metrics["teacher_proto_mode"] = self.teacher_proto_mode
        metrics["n_replay"] = int(self._n_replay_used)
        metrics["n_neighbor_replay"] = int(self._n_neighbor_used)
        metrics["n_forgotten_replay"] = int(self._n_forgotten_used)
        metrics["n_teacher_proto"] = int(self._n_teacher_proto)
        metrics["memory_segments"] = len(self._memories)
        metrics["train.n_steps"] = int(metrics.get("batches", 0))
        return metrics

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

    def _mix_current_and_replay(self, segment: Segment) -> List[Example]:
        current = list(segment.train)
        if not self._memories or self.replay_ratio <= 0:
            return current
        n = len(current)
        if self.strategy == "uniform":
            n_replay = max(0, int(round(n * self.replay_ratio)))
            pool = [ex for mem in self._memories for ex in mem.examples]
            replay = _sample_examples(pool, n_replay)
            self._n_replay_used = len(replay)
        else:
            n_nb = max(0, int(round(n * self.neighbor_ratio)))
            n_fg = max(0, int(round(n * self.forgotten_ratio)))
            neighbors = self._select_neighbors(n_nb)
            forgotten = self._select_forgotten(n_fg)
            replay = neighbors + forgotten
            self._n_neighbor_used = len(neighbors)
            self._n_forgotten_used = len(forgotten)
            self._n_replay_used = len(replay)
        mixed = current + replay
        random.shuffle(mixed)
        return mixed

    def _select_neighbors(self, n: int) -> List[Example]:
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
            return _sample_examples(pool, n)
        ranked.sort(key=lambda x: x[0], reverse=True)
        pool: List[Example] = []
        for _, mem in ranked:
            pool.extend(mem.examples)
        return _sample_examples(pool, n)

    def _select_forgotten(self, n: int) -> List[Example]:
        if n <= 0 or not self._memories:
            return []
        best_sid = None
        best_f = -1.0
        known = {mem.segment_id for mem in self._memories}
        for sid, fgt in self._forgetting_by_segment.items():
            if sid in known and float(fgt) > best_f:
                best_f = float(fgt)
                best_sid = int(sid)
        if best_sid is None or best_f <= 0:
            pool = [ex for mem in self._memories for ex in mem.examples]
            return _sample_examples(pool, n)
        for mem in self._memories:
            if mem.segment_id == best_sid:
                return _sample_examples(mem.examples, n)
        pool = [ex for mem in self._memories for ex in mem.examples]
        return _sample_examples(pool, n)

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

    def _build_memory(self, segment: Segment, model: Any, lora: Any) -> SegmentMemory:
        instruction = str(segment.train[0].instruction) if segment.train else str(segment.segment_name)
        score_types: List[str] = []
        gold_lens: List[float] = []
        for ex in segment.train:
            fmt = _example_format(ex)
            score_types.append(fmt["task_score_type"])
            gold_lens.append(fmt["mean_gold_len"])
        task_score_type = _majority(score_types) if score_types else "strict_em"
        mean_gold_len = sum(gold_lens) / max(1, len(gold_lens))
        embedding = _frozen_instruction_embed(model, lora, instruction)
        examples = _pick_diverse_k(segment.train, self.buffer_k)
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

    def _example_to_proto(self, ex: Example, mem: Optional[SegmentMemory]) -> Dict[str, Any]:
        golds = _example_golds(ex)
        return {
            "instruction": ex.instruction,
            "input": ex.input,
            "output": golds[0] if golds else "",
            "task_score_type": mem.task_score_type if mem else _example_format(ex)["task_score_type"],
            "mean_gold_len": mem.mean_gold_len if mem else _example_format(ex)["mean_gold_len"],
        }


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
