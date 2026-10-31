#!/usr/bin/env python3
"""Serial ACL pending-extension queue. One training process. Never overwrites a run."""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

ROOT = Path("/root/research-repos/opd-lora_v3")
sys.path.insert(0, str(ROOT / "scripts"))
import append_pending_results as pending  # noqa: E402
import run_v3_acl_queue as acl  # noqa: E402

GROUPS = pending.GROUPS
ORDER = ["e2", "e4lam", "e4off", "e6", "e3", "e9", "e10", "e11"]
BLOCKED = [
    {
        "name": "v3_sft_then_opsd",
        "group": "e3",
        "status": "blocked",
        "reason": (
            "No legal Sequential/SFT adapter on v3 protocol. "
            "v2pre_citb_sequential_* used processed_stream_file without *_multiref.json "
            "and snapshots have no eval_normalization.format_normalize."
        ),
    },
    {
        "name": "opsd_temperature_sweep",
        "group": "e4",
        "status": "blocked",
        "reason": "JSD has no separate distillation T; rollout_temperature stays 1.0. Not in the original objective.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group",
        default="all",
        help="comma list of e2,e4lam,e4off,e6,e3,e9,e10,e11 or all",
    )
    args = parser.parse_args()
    acl.setup_env()
    acl.LOGDIR.mkdir(parents=True, exist_ok=True)
    lock_handle = acl.LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another v3 queue holds /tmp/rp_lora_v3_gpu.lock", flush=True)
        raise SystemExit(2)

    wanted = ORDER if args.group == "all" else [g.strip() for g in args.group.split(",") if g.strip()]
    for g in wanted:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}")

    summary = acl.load_summary()
    summary["pending_status"] = "running"
    summary["blocked"] = BLOCKED
    acl.save_summary(summary)
    for row in BLOCKED:
        acl.record(row["name"], "BLOCKED", row["reason"])
    try:
        for group in wanted:
            acl.record(group, "GROUP_START", pending.TITLES[group])
            for name in GROUPS[group]:
                acl.run_one(name, summary)
            pending.main()
            acl.record(group, "GROUP_DONE", pending.TITLES[group])
        summary = acl.load_summary()
        failed = [row["name"] for row in summary["runs"] if not row.get("ok") and not row.get("skipped")]
        summary["pending_status"] = "failed" if failed else "ok"
        summary["pending_failed"] = failed
        acl.save_summary(summary)
        print(json.dumps({"status": summary["pending_status"], "failed": failed, "blocked": BLOCKED}, indent=2), flush=True)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
