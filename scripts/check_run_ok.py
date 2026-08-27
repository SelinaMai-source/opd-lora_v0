#!/usr/bin/env python3
"""Validate a training run: exit 0, non-NaN loss, eval on disk."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RESULTS = Path("/root/autodl-tmp/rp_lora_v0/results")
LOGS = Path("/root/autodl-tmp/rp_lora_v0/logs")


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


def _find_run_dir(experiment_name: str, log_text: str) -> Path | None:
    for line in log_text.splitlines():
        if "Run ID:" in line:
            run_id = line.split("Run ID:", 1)[1].strip()
            cand = RESULTS / "runs" / run_id
            if cand.is_dir():
                return cand
    runs = RESULTS / "runs"
    if not runs.is_dir():
        return None
    matches = sorted(
        [p for p in runs.iterdir() if p.is_dir() and experiment_name in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: check_run_ok.py <log_name> <experiment_name>")
        return 2
    log_name = sys.argv[1]
    experiment_name = sys.argv[2]
    exit_path = LOGS / f"{log_name}.exitcode"
    log_path = LOGS / f"{log_name}.log"
    if not exit_path.exists():
        print(f"FAIL: missing {exit_path}")
        return 1
    ec = int(exit_path.read_text().strip() or "1")
    if ec != 0:
        print(f"FAIL: exit_code={ec}")
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:] if log_path.exists() else ""
        print(tail)
        return 1
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    run_dir = _find_run_dir(experiment_name, log_text)
    if run_dir is None:
        print(f"FAIL: cannot find run dir for {experiment_name}")
        return 1
    metrics_path = run_dir / "final_metrics.json"
    if not metrics_path.exists():
        print(f"FAIL: missing {metrics_path}")
        return 1
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final = metrics.get("final") or {}
    loss_keys = [k for k in final if "loss" in k.lower()]
    nan_losses = [k for k in loss_keys if _is_nan(final.get(k))]
    if nan_losses:
        print(f"FAIL: NaN loss fields {nan_losses} in {metrics_path}")
        return 1
    # also scan train_metrics.json under segments
    seg_dirs = sorted(run_dir.glob("segment_*"))
    if not seg_dirs:
        print(f"FAIL: no segment_* dirs in {run_dir}")
        return 1
    eval_ok = 0
    for sd in seg_dirs:
        tm = sd / "train_metrics.json"
        em = sd / "eval_metrics.json"
        if tm.exists():
            tmj = json.loads(tm.read_text(encoding="utf-8"))
            for k, v in tmj.items():
                if "loss" in str(k).lower() and _is_nan(v):
                    print(f"FAIL: NaN {k} in {tm}")
                    return 1
        if not em.exists():
            print(f"FAIL: missing eval {em}")
            return 1
        eval_ok += 1
    csv_path = RESULTS / "tables" / f"{run_dir.name}_segment_metrics.csv"
    extra = f" csv={csv_path.exists()}"
    print(
        f"OK run_dir={run_dir} segments={len(seg_dirs)} eval_files={eval_ok} "
        f"loss_keys={loss_keys}{extra}"
    )
    seen = final.get("eval.seen_avg_score")
    forget = final.get("eval.forgetting")
    print(f"  seen_avg_score={seen} forgetting={forget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
