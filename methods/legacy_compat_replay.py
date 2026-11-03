"""Legacy uniform and proto replay.

Quota stays ``methods.pgdr.quota`` (Python round). Forty-five current rows
therefore still request 9 uniform rows, or 4+4 proto rows. This module does
not call the mixed-batch allocator and does not relabel those runs as v3c.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod
from methods.pgdr import quota, replay_quota

LEGACY_STRATEGIES = ("uniform", "proto")


def legacy_counts(n_current: int, *, strategy: str, replay_ratio: float = 0.2, neighbor_ratio: float = 0.1, forgotten_ratio: float = 0.1) -> Dict[str, int]:
    """Return the old per-slot counts. Seed is not an argument."""
    return replay_quota(
        int(n_current),
        strategy=str(strategy),
        replay_ratio=float(replay_ratio),
        neighbor_ratio=float(neighbor_ratio),
        forgotten_ratio=float(forgotten_ratio),
    )


def build(cfg: Dict[str, Any]) -> CeOpsdReplayMethod:
    """Construct the existing replay class on the legacy branch."""
    built = copy.deepcopy(cfg)
    replay = dict(built.get("replay") or {})
    strategy = str(replay.get("strategy", "proto")).strip().lower()
    if strategy not in LEGACY_STRATEGIES:
        raise ValueError(f"legacy replay only accepts {LEGACY_STRATEGIES}, got {strategy}")
    replay["strategy"] = strategy
    replay["budget_policy"] = "legacy"
    built["replay"] = replay
    return CeOpsdReplayMethod(built)


__all__ = ["LEGACY_STRATEGIES", "build", "legacy_counts", "quota", "replay_quota"]
