#!/usr/bin/env python3
"""Confirm a v3h smoke run did not train on the holdout or select from the eval split."""
from __future__ import annotations

import json
import sys
from pathlib import Path

RUNS = Path("/root/autodl-tmp/rp_lora_v3_runs/runs")
BENCH = Path("/root/research-repos/opd-lora_v3/benchmark/instrdialog/citb_cl_dialogue_tasks_train50_eval10_multiref.json")
DEFAULT_RUN = "v3h_proto_opsd_instrdialog_seed456_smoke2seg"


def key(row: dict) -> tuple:
    return (str(row.get("instruction", "")), str(row.get("input", "")))


def main() -> int:
    run_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    run_dir = RUNS / run_name
    metrics_path = run_dir / "final_metrics.json"
    failures = []
    if not metrics_path.is_file():
        failures.append(f"missing {metrics_path}")
        _emit(run_dir, failures)
        return 2
    stream = json.loads(BENCH.read_text(encoding="utf-8"))["stream"]
    seg_dirs = sorted(p for p in run_dir.glob("segment_*") if p.is_dir())
    if len(seg_dirs) < 2:
        failures.append(f"expected 2 segment dirs, found {len(seg_dirs)}")
    for seg_dir in seg_dirs:
        payload = json.loads((seg_dir / "train_metrics.json").read_text(encoding="utf-8"))
        sid = int(payload.get("holdout_audit", {}).get("segment_id", seg_dir.name.split("_")[-1]))
        audit = payload.get("holdout_audit") or {}
        if payload.get("monitor") != "train_holdout":
            failures.append(f"{seg_dir.name}: monitor={payload.get('monitor')}")
        if payload.get("selector_version") != "train_holdout":
            failures.append(f"{seg_dir.name}: selector_version={payload.get('selector_version')}")
        if payload.get("holdout_in_train") is not False:
            failures.append(f"{seg_dir.name}: holdout_in_train={payload.get('holdout_in_train')}")
        if payload.get("holdout_in_memory") is not False:
            failures.append(f"{seg_dir.name}: holdout_in_memory={payload.get('holdout_in_memory')}")
        if payload.get("eval_split_read_for_selection") is not False:
            failures.append(f"{seg_dir.name}: eval_split_read_for_selection")
        if payload.get("on_eval_end_ignored") is not True:
            failures.append(f"{seg_dir.name}: on_eval_end_ignored={payload.get('on_eval_end_ignored')}")
        hold_keys = {tuple(pair) for pair in audit.get("holdout_keys") or []}
        if len(hold_keys) != 5:
            failures.append(f"{seg_dir.name}: holdout_n={len(hold_keys)}")
        raw = stream[sid]
        train_keys = {key(row) for row in raw["train"]}
        eval_keys = {key(row) for row in raw["eval"]}
        if not hold_keys <= train_keys:
            failures.append(f"{seg_dir.name}: holdout is not a subset of train")
        if hold_keys & eval_keys:
            failures.append(f"{seg_dir.name}: holdout intersects official eval")
        if int(audit.get("fit_n", -1)) != 45:
            failures.append(f"{seg_dir.name}: fit_n={audit.get('fit_n')}")
    _emit(run_dir, failures)
    if failures:
        print("SMOKE_LEAKAGE_OR_AUDIT_FAIL")
        for item in failures:
            print(item)
        return 1
    print("SMOKE_HOLDOUT_OK")
    return 0


def _emit(run_dir: Path, failures: list) -> None:
    out = Path("/root/autodl-tmp/rp_lora_v3_runs/smoke_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"run_dir": str(run_dir), "ok": not failures, "failures": failures}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
