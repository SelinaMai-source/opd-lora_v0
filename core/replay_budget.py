"""Unified mixed-batch replay budget and explicit L2 cosine.

The legacy ``uniform`` / ``proto`` quotas stay in ``methods.pgdr.quota``
(Python ``round``, so ``quota(45, 0.1) + quota(45, 0.1) = 8``). This module
is the corrected definition used only by ``random``, ``compat_uniform`` and
``true_prototype``.

Mixed-batch: ``n_replay = half_up(ρ / (1 - ρ) * n_current)`` with ρ = 0.20.
45 current rows produce 11 replay rows. The batch is 56, so the realized
fraction is 11/56 ≈ 19.64%, not a claim of exactly 20%.

Related and Degraded split that single integer with the largest-remainder
method. Equal remainders go to Related first, so 11 becomes 6 + 5. Random,
compatibility-uniform and true prototype all receive the same ``n_replay``.
The training seed is not an input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def half_up(value: float) -> int:
    """Round half away from zero for nonnegative numbers (not banker's round)."""
    number = _as_fraction(value)
    if number < 0:
        raise ValueError(f"half_up expects a nonnegative value, got {value}")
    return int(math.floor(number + Fraction(1, 2)))


def mixed_batch_n_replay(n_current: int, rho: float) -> int:
    """How many replay rows a mixed batch needs so replay is about ``rho`` of the batch."""
    n_current = int(n_current)
    if n_current < 0:
        raise ValueError(f"n_current must be >= 0, got {n_current}")
    rho_q = _rho_fraction(rho)
    if n_current == 0 or rho_q == 0:
        return 0
    raw = (rho_q / (1 - rho_q)) * n_current
    return half_up(raw)


def realized_replay_fraction(n_replay: int, n_current: int) -> float:
    """Share of the mixed batch that is replay. Zero when both counts are zero."""
    n_replay = int(n_replay)
    n_current = int(n_current)
    denom = n_replay + n_current
    if denom <= 0:
        return 0.0
    return float(n_replay) / float(denom)


def largest_remainder(total: int, names: Sequence[str], weights: Optional[Sequence[float]] = None) -> Dict[str, int]:
    """Hamilton apportionment. Earlier ``names`` win when fractional parts tie.

    The returned values are nonnegative integers and sum to ``total``.
    """
    total = int(total)
    if total < 0:
        raise ValueError(f"total must be >= 0, got {total}")
    labels = [str(name) for name in names]
    if len(labels) != len(set(labels)):
        raise ValueError(f"part names must be unique, got {labels}")
    if weights is None:
        weights = [1.0] * len(labels)
    if len(weights) != len(labels):
        raise ValueError("weights and names must have the same length")
    if not labels:
        if total != 0:
            raise ValueError("cannot allocate a positive total across no parts")
        return {}
    weight_values = [float(w) for w in weights]
    if any(w < 0 for w in weight_values):
        raise ValueError(f"weights must be nonnegative, got {weight_values}")
    weight_sum = sum(weight_values)
    if weight_sum <= 0:
        raise ValueError("at least one weight must be positive")
    if total == 0:
        return {label: 0 for label in labels}
    if _weights_equal(weight_values):
        base, extra = divmod(total, len(labels))
        out = {label: base for label in labels}
        for label in labels[:extra]:
            out[label] += 1
        return out
    raw = [total * (w / weight_sum) for w in weight_values]
    floors = [int(math.floor(x + 1e-12)) for x in raw]
    leftover = total - sum(floors)
    order = sorted(range(len(labels)), key=lambda i: (-(raw[i] - floors[i]), i))
    counts = floors[:]
    for rank in range(leftover):
        counts[order[rank]] += 1
    return {labels[i]: int(counts[i]) for i in range(len(labels))}


@dataclass(frozen=True)
class MixedBatchBudget:
    n_current: int
    rho: float
    n_replay_target: int
    n_related: int
    n_degraded: int
    realized_replay_fraction: float
    ratio_definition: str = "mixed_batch"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_current": int(self.n_current),
            "rho": float(self.rho),
            "n_replay_target": int(self.n_replay_target),
            "n_related": int(self.n_related),
            "n_degraded": int(self.n_degraded),
            "realized_replay_fraction": float(self.realized_replay_fraction),
            "ratio_definition": str(self.ratio_definition),
        }


def allocate_mixed_batch(
    n_current: int,
    rho: float = 0.20,
    *,
    parts: Sequence[str] = ("related", "degraded"),
) -> MixedBatchBudget:
    """One replay count, then a largest-remainder split. Seed is not an argument."""
    n_target = mixed_batch_n_replay(n_current, rho)
    split = largest_remainder(n_target, parts)
    if sum(split.values()) != n_target:
        raise RuntimeError(f"largest remainder drifted: {split} vs {n_target}")
    return MixedBatchBudget(
        n_current=int(n_current),
        rho=float(rho),
        n_replay_target=int(n_target),
        n_related=int(split.get("related", 0)),
        n_degraded=int(split.get("degraded", 0)),
        realized_replay_fraction=realized_replay_fraction(n_target, int(n_current)),
        ratio_definition="mixed_batch",
    )


def vector_l2(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in values))


def l2_normalize(values: Sequence[float]) -> List[float]:
    norm = vector_l2(values)
    if norm <= 0.0:
        return [0.0 for _ in values]
    return [float(x) / norm for x in values]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine after explicit L2 normalization of each argument.

    A raw dot product is not used. Vectors that are already unit length stay
    unit length. Unequal widths use the shared prefix after each full-vector
    normalization.
    """
    if not left or not right:
        return 0.0
    aa = l2_normalize(left)
    bb = l2_normalize(right)
    width = min(len(aa), len(bb))
    return float(sum(aa[i] * bb[i] for i in range(width)))


def rank_segment_ids(
    scores: Sequence[Tuple[int, float]],
) -> List[int]:
    """Highest cosine first. Equal cosines break toward the smaller segment id."""
    ordered = sorted(((int(sid), float(score)) for sid, score in scores), key=lambda row: (-row[1], row[0]))
    return [sid for sid, _score in ordered]


def validate_v3c_replay_block(replay: Mapping[str, Any]) -> None:
    """Paper entry check. Missing keys are a refusal, not a silent default."""
    required = (
        "monitor",
        "holdout_seed",
        "holdout_n",
        "budget_policy",
        "ratio_definition",
        "target_fraction",
    )
    missing = [key for key in required if key not in replay]
    problems: List[str] = list(missing)
    if "monitor" in replay and str(replay.get("monitor", "")).strip().lower() != "train_holdout":
        problems.append("monitor!=train_holdout")
    if "holdout_seed" in replay and int(replay.get("holdout_seed")) != 456:
        problems.append("holdout_seed!=456")
    if "holdout_n" in replay and int(replay.get("holdout_n")) != 5:
        problems.append("holdout_n!=5")
    if "budget_policy" in replay and str(replay.get("budget_policy", "")).strip().lower() != "unified":
        problems.append("budget_policy!=unified")
    if "ratio_definition" in replay and str(replay.get("ratio_definition", "")).strip().lower() != "mixed_batch":
        problems.append("ratio_definition!=mixed_batch")
    if "target_fraction" in replay and not math.isclose(float(replay.get("target_fraction")), 0.20, rel_tol=0.0, abs_tol=1e-12):
        problems.append("target_fraction!=0.20")
    strategy = str(replay.get("strategy", "")).strip().lower()
    if strategy not in {"random", "compat_uniform", "true_prototype"}:
        problems.append(f"strategy={strategy or 'missing'}")
    if problems:
        raise ValueError("v3c config refused: " + ", ".join(problems))


def _rho_fraction(rho: float) -> Fraction:
    number = float(rho)
    if math.isnan(number) or math.isinf(number) or number < 0.0 or number >= 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    return Fraction(number).limit_denominator(1_000_000)


def _as_fraction(value: float) -> Fraction:
    if isinstance(value, Fraction):
        return value
    number = float(value)
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"value must be finite, got {value}")
    return Fraction(number).limit_denominator(1_000_000)


def _weights_equal(weights: Sequence[float]) -> bool:
    head = float(weights[0])
    return all(abs(float(w) - head) <= 1e-12 for w in weights)
