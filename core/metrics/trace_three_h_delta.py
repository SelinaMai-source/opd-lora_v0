"""
TRACE benchmark: General / Instruction / Safety (3H) delta metrics — STUB.

Full implementation requires:
  1. Pre-training baseline scores on held-out probes (MMLU subset, IFEval, safety suite)
  2. Post-continual-learning re-evaluation on the same probes
  3. Delta = post - pre per dimension

Reference: TRACE paper (arXiv:2310.06762) and external_baselines/trace_rcl/inference/.

Until raw TRACE data and probe suites are wired, callers receive ``None`` placeholders
and a documented skip reason — results must NOT be fabricated in main tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TraceThreeHDelta:
    general_ability_delta: Optional[float]
    instruction_following_delta: Optional[float]
    safety_delta: Optional[float]
    status: str
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_general_ability_delta": self.general_ability_delta,
            "trace_instruction_following_delta": self.instruction_following_delta,
            "trace_safety_delta": self.safety_delta,
            "trace_3h_delta_status": self.status,
            "trace_3h_delta_notes": self.notes,
        }


def compute_trace_three_h_delta(
    run_dir: str,
    *,
    baseline_scores: Optional[Dict[str, float]] = None,
) -> TraceThreeHDelta:
    """Return stub 3H delta; implement probe re-eval when TRACE full pipeline is ready."""
    _ = run_dir, baseline_scores
    return TraceThreeHDelta(
        general_ability_delta=None,
        instruction_following_delta=None,
        safety_delta=None,
        status="not_implemented",
        notes=(
            "3H delta stub: needs pre/post probe evaluation (MMLU/IFEval/safety). "
            "See configs/paper/published_setting/README.md and TRACE raw download."
        ),
    )
