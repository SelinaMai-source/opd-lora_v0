#!/usr/bin/env python3
"""Budget and retrieval smoke that does not load an 8B checkpoint."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod, SegmentMemory  # noqa: E402
from core.data import Example, Segment  # noqa: E402
from core.replay_budget import allocate_mixed_batch, cosine_similarity  # noqa: E402


def _ex(seg: int, i: int) -> Example:
    return Example(f"task-{seg}", f"row-{i}", f"gold-{seg}-{i}", [f"gold-{seg}-{i}"])


def _unit(value: float) -> list:
    return [value, math.sqrt(max(0.0, 1.0 - value * value))]


def main() -> None:
    budget = allocate_mixed_batch(45, 0.20)
    if budget.n_replay_target != 11 or budget.n_related != 6 or budget.n_degraded != 5:
        raise SystemExit(f"budget {budget}")
    if abs(budget.realized_replay_fraction - (11 / 56)) > 1e-12:
        raise SystemExit("realized fraction is not 11/56")
    current_vec = [1.0, 0.0]
    memories = [
        SegmentMemory(1, "B", "strict_em", 1.0, [v * 50 for v in _unit(0.40)], [_ex(1, i) for i in range(4)]),
        SegmentMemory(2, "C", "strict_em", 1.0, [v * 3 for v in _unit(-0.20)], [_ex(2, i) for i in range(4)]),
        SegmentMemory(10, "A", "strict_em", 1.0, [v * 0.01 for v in _unit(0.90)], [_ex(10, i) for i in range(4)]),
    ]
    if abs(cosine_similarity(current_vec, memories[2].embedding) - 0.90) > 1e-9:
        raise SystemExit("cosine helper drifted")
    cfg = {
        "seed": 456,
        "replay": {
            "strategy": "true_prototype",
            "replay_ratio": 0.2,
            "buffer_k": 10,
            "monitor": "train_holdout",
            "holdout_n": 5,
            "holdout_seed": 456,
            "budget_policy": "unified",
            "ratio_definition": "mixed_batch",
            "target_fraction": 0.20,
        },
        "teacher_proto": {"mode": "none"},
        "ce_opsd": {"lambda_opsd": 0.0},
    }
    proto = CeOpsdReplayMethod(cfg)
    proto._current_memory = SegmentMemory(8, "cur", "strict_em", 1.0, current_vec, [])
    proto._memories = memories
    proto._forgetting_by_segment = {1: 0.2, 2: 0.2, 10: 0.2}
    random.seed(0)
    proto._mix_current_and_replay(Segment(8, "cur", [_ex(8, i) for i in range(4)], []))
    if proto._last_replay_log["related_source_segment_ids"] != [10]:
        raise SystemExit(f"true_prototype did not take A: {proto._last_replay_log}")
    cfg_u = dict(cfg)
    cfg_u["replay"] = dict(cfg["replay"])
    cfg_u["replay"]["strategy"] = "compat_uniform"
    seen = set()
    for seed in range(12):
        random.seed(seed)
        uniform = CeOpsdReplayMethod(cfg_u)
        uniform._current_memory = proto._current_memory
        uniform._memories = memories
        uniform._forgetting_by_segment = proto._forgetting_by_segment
        uniform._mix_current_and_replay(Segment(8, "cur", [_ex(8, i) for i in range(4)], []))
        if "cosine prototype" in uniform._last_replay_log["selector"] or uniform._last_replay_log["selector"] == "l2_cosine_descending":
            raise SystemExit("compat_uniform was labeled as cosine prototype")
        seen.add(uniform._last_replay_log["related_source_segment_ids"][0])
    if seen == {10}:
        raise SystemExit("compat_uniform matched true_prototype on every draw")
    from baselines.wrappers import REGISTRY
    from benchmarks.instrdialog import get_metadata as id_meta
    from benchmarks.instrdialogpp import get_metadata as pp_meta
    from methods.selectors import budget_for

    if len(REGISTRY) != 14:
        raise SystemExit(f"expected 14 real wrappers, got {len(REGISTRY)}")
    if budget_for(45).n_replay_target != 11:
        raise SystemExit("public allocator drifted")
    if "/root/autodl-tmp" in str(id_meta()) or "/root/autodl-tmp" in str(pp_meta()):
        raise SystemExit("benchmark metadata requires a private disk path")
    print("SMOKE_BUDGET_OK", flush=True)


if __name__ == "__main__":
    main()
