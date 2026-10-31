"""PGDR memory, neighbor slot, forgotten slot, and replay quota.

Two forgotten-selector versions. Call sites must pass ``selector_version``:

- ``eval_split_monitor``: scores come from official eval forgetting written by
  ``on_eval_end`` (old protocol). The selector still does not read future tasks,
  but it does use test-split correctness.
- ``train_holdout``: scores come only from a fixed train-set holdout. The
  official eval split is not an input to selection.

Neighbor ranking stays frozen-instruction cosine. Quota math is shared.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.data import Example, Segment

MONITOR_EVAL_SPLIT = "eval_split_monitor"
MONITOR_TRAIN_HOLDOUT = "train_holdout"
SELECTOR_VERSIONS = (MONITOR_EVAL_SPLIT, MONITOR_TRAIN_HOLDOUT)


def quota(n: int, ratio: float) -> int:
    """Same rounding the old replay mixer uses: ``max(0, int(round(n * ratio)))``."""
    return max(0, int(round(float(n) * float(ratio))))


def replay_quota(
    n_current: int,
    *,
    strategy: str,
    replay_ratio: float,
    neighbor_ratio: float,
    forgotten_ratio: float,
) -> Dict[str, int]:
    strategy = str(strategy).strip().lower()
    if strategy == "uniform":
        n_replay = quota(n_current, replay_ratio)
        return {
            "n_replay": n_replay,
            "n_uniform": n_replay,
            "n_neighbor": 0,
            "n_forgotten": 0,
        }
    if strategy != "proto":
        raise ValueError(f"replay.strategy must be uniform or proto, got: {strategy}")
    n_nb = quota(n_current, neighbor_ratio)
    n_fg = quota(n_current, forgotten_ratio)
    return {
        "n_replay": n_nb + n_fg,
        "n_uniform": 0,
        "n_neighbor": n_nb,
        "n_forgotten": n_fg,
    }


def example_key(ex: Example) -> Tuple[str, str]:
    return (str(ex.instruction), str(ex.input))


def split_train_holdout(
    examples: Sequence[Example],
    *,
    segment_id: int,
    seed: int,
    holdout_n: int = 5,
) -> Tuple[List[Example], List[Example], List[int]]:
    """Hold out ``holdout_n`` train rows. Deterministic in ``(seed, segment_id)``.

    The RNG is local, so the split does not consume the training RNG and is
    identical for every method that shares ``seed`` and segment order. At least
    one row stays in the fit set when the segment is non-empty.
    """
    pool = list(examples)
    n = len(pool)
    if n <= 1:
        return pool, [], []
    k = min(max(0, int(holdout_n)), n - 1)
    rng = random.Random((int(seed) * 1_000_003) ^ (int(segment_id) * 9_176) ^ 0x5A17)
    order = list(range(n))
    rng.shuffle(order)
    hold_idx = sorted(order[:k])
    hold_set = set(hold_idx)
    holdout = [pool[i] for i in range(n) if i in hold_set]
    fit = [pool[i] for i in range(n) if i not in hold_set]
    return fit, holdout, hold_idx


def forgetting_from_accuracies(
    per_seg_acc: Sequence[Tuple[int, float]],
    historical_best: Optional[Dict[int, float]] = None,
) -> Tuple[List[Dict[str, float]], float, Dict[int, float]]:
    """Same forgetting definition as ``core.evaluate`` (lower is better).

    The last row is the current segment and contributes 0. Earlier segments
    contribute ``max(0, best_before - acc)``. ``historical_best`` is the best
    accuracy seen before this call and is not mutated; the returned dict is
    the updated best.
    """
    best = {int(k): float(v) for k, v in (historical_best or {}).items()}
    rows_in = [(int(sid), float(acc)) for sid, acc in per_seg_acc]
    previous_ids = {sid for sid, _ in rows_in[:-1]}
    rows: List[Dict[str, float]] = []
    values: List[float] = []
    for sid, acc in rows_in:
        best_before = float(best.get(sid, acc))
        seg_forgetting = float(max(0.0, best_before - acc)) if sid in previous_ids else 0.0
        if sid in previous_ids:
            values.append(seg_forgetting)
        rows.append(
            {
                "segment_id": int(sid),
                "accuracy": float(acc),
                "best_historical_accuracy": float(best_before),
                "forgetting": float(seg_forgetting),
            }
        )
        best[sid] = max(best_before, float(acc))
    mean = float(sum(values) / len(values)) if values else 0.0
    return rows, mean, best


def select_forgotten(
    memories: Sequence[Any],
    forgetting_by_segment: Dict[int, float],
    n: int,
    *,
    selector_version: str,
    sample_fn=None,
) -> List[Example]:
    """Most-forgotten segment's buffer, else uniform over memory.

    ``selector_version`` does not change the ranking. It is required so the
    caller records whether ``forgetting_by_segment`` came from the eval split
    or the train holdout. Sampling uses the global ``random`` module, matching
    the old selector, unless ``sample_fn`` is injected.
    """
    if selector_version not in SELECTOR_VERSIONS:
        raise ValueError(
            f"selector_version must be one of {SELECTOR_VERSIONS}, got: {selector_version}"
        )
    draw = sample_fn or _sample_examples
    if n <= 0 or not memories:
        return []
    best_sid = None
    best_f = -1.0
    known = {int(mem.segment_id) for mem in memories}
    for sid, fgt in forgetting_by_segment.items():
        if int(sid) in known and float(fgt) > best_f:
            best_f = float(fgt)
            best_sid = int(sid)
    if best_sid is None or best_f <= 0:
        pool = [ex for mem in memories for ex in mem.examples]
        return draw(pool, n)
    for mem in memories:
        if int(mem.segment_id) == best_sid:
            return draw(list(mem.examples), n)
    pool = [ex for mem in memories for ex in mem.examples]
    return draw(pool, n)


def score_holdout_map(
    model: Any,
    holdouts: Dict[int, Sequence[Example]],
    *,
    normalization_cfg: Optional[Dict[str, Any]] = None,
    max_new_tokens: int = 64,
) -> List[Tuple[int, float]]:
    """Accuracy of each segment's train holdout. Does not take an eval split.

    Each holdout is scored by itself (batch = holdout size, typically 5) via
    the same ``_eval_segment`` match rate used for official eval. The examples
    passed in are the only rows generated.
    """
    from core.evaluate import _eval_segment

    cfg = dict(normalization_cfg or {})
    accs: List[Tuple[int, float]] = []
    for sid in sorted(int(k) for k in holdouts):
        examples = list(holdouts[sid])
        if not examples:
            accs.append((sid, 0.0))
            continue
        synthetic = Segment(
            segment_id=int(sid),
            segment_name=f"train_holdout_{sid}",
            train=[],
            eval=examples,
        )
        acc, _task_aware, _routing, _rows = _eval_segment(
            model=model,
            segment=synthetic,
            max_new_tokens=int(max_new_tokens),
            router=None,
            lora_bank=None,
            segment_id=int(sid),
            normalization_cfg=cfg,
        )
        accs.append((int(sid), float(acc)))
    return accs


def keys_overlap(left: Iterable[Example], right: Iterable[Example]) -> bool:
    return bool({example_key(ex) for ex in left} & {example_key(ex) for ex in right})


def _sample_examples(pool: List[Example], n: int) -> List[Example]:
    if n <= 0 or not pool:
        return []
    if n >= len(pool):
        return list(pool)
    return random.sample(pool, k=n)
