"""Corrected selectors. The count comes from ``core.replay_budget``.

Sampling stays inside ``CeOpsdReplayMethod``. ``compat_uniform`` is not a
cosine prototype. ``true_prototype`` is the only selector that fills Related
by explicit L2 cosine.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod
from core.replay_budget import allocate_mixed_batch, validate_v3c_replay_block

SELECTORS = ("random", "compat_uniform", "true_prototype")


def budget_for(n_current: int, rho: float = 0.20):
    """Shared mixed-batch count. Random, compat_uniform and true_prototype all use it."""
    return allocate_mixed_batch(int(n_current), float(rho))


def build(cfg: Dict[str, Any], strategy: str) -> CeOpsdReplayMethod:
    """Construct the existing replay class with one corrected selector."""
    name = str(strategy).strip().lower()
    if name not in SELECTORS:
        raise ValueError(f"selector must be one of {SELECTORS}, got {strategy}")
    built = copy.deepcopy(cfg)
    replay = dict(built.get("replay") or {})
    replay["strategy"] = name
    replay["budget_policy"] = "unified"
    replay["ratio_definition"] = "mixed_batch"
    replay["target_fraction"] = 0.20
    replay["monitor"] = "train_holdout"
    replay["holdout_seed"] = 456
    replay.setdefault("holdout_n", 5)
    replay.setdefault("replay_ratio", 0.2)
    validate_v3c_replay_block(replay)
    built["replay"] = replay
    return CeOpsdReplayMethod(built)


__all__ = ["SELECTORS", "budget_for", "build"]
