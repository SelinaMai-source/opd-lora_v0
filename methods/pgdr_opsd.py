"""CE plus optional OPSD on a corrected selector.

The loss is still ``methods.opsd.ce_plus_opsd`` inside the existing trainer.
``lambda_opsd <= 0`` leaves CE unchanged. This module does not add a second loss.
"""
from __future__ import annotations

import copy
from typing import Any, Dict

from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod
from methods.opsd import ce_plus_opsd
from methods.selectors import SELECTORS, budget_for, build as build_selector

__all__ = ["SELECTORS", "budget_for", "build", "ce_plus_opsd"]


def build(cfg: Dict[str, Any], strategy: str = "true_prototype", lambda_opsd: float = 0.2) -> CeOpsdReplayMethod:
    built = copy.deepcopy(cfg)
    block = dict(built.get("ce_opsd") or {})
    block["lambda_opsd"] = float(lambda_opsd)
    built["ce_opsd"] = block
    return build_selector(built, strategy)
