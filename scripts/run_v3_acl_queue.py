#!/usr/bin/env python3
"""Serial ACL v3 queue. One training process. Never overwrites a run directory.

Order: v3h PGDR+CE+OPSD 2-seg smoke, then 12 old-protocol seed 789/42 runs,
then 8 v3h seed-456 runs if the smoke audit passes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path("/root/research-repos/opd-lora_v3")
PYTHON = "/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python"
RUNS_ROOT = Path("/root/autodl-tmp/rp_lora_v3_runs")
RESULTS = RUNS_ROOT / "runs"
LOGDIR = RUNS_ROOT / "logs"
SUMMARY = LOGDIR / "acl_queue_summary.json"
LOCK = Path("/tmp/rp_lora_v3_gpu.lock")
sys.path.insert(0, str(ROOT))
from core.run_guard import explain_exit, fingerprints_match, load_snapshot  # noqa: E402

SMOKE = "v3h_proto_opsd_instrdialog_seed456_smoke2seg"
OLD_12 = [
    "v3_rand_ce_instrdialog_seed789",
    "v3_rand_ce_instrdialog_seed42",
    "v3_proto_ce_instrdialog_seed789",
    "v3_proto_ce_instrdialog_seed42",
    "v3_rand_opsd_instrdialog_seed789",
    "v3_rand_opsd_instrdialog_seed42",
    "v3_rand_ce_instrdialogpp_seed789",
    "v3_rand_ce_instrdialogpp_seed42",
    "v3_proto_ce_instrdialogpp_seed789",
    "v3_proto_ce_instrdialogpp_seed42",
    "v3_rand_opsd_instrdialogpp_seed789",
    "v3_rand_opsd_instrdialogpp_seed42",
]
HOLDOUT_8 = [
    "v3h_rand_ce_instrdialog_seed456",
    "v3h_proto_ce_instrdialog_seed456",
    "v3h_rand_opsd_instrdialog_seed456",
    "v3h_proto_opsd_instrdialog_seed456",
    "v3h_rand_ce_instrdialogpp_seed456",
    "v3h_proto_ce_instrdialogpp_seed456",
    "v3h_rand_opsd_instrdialogpp_seed456",
    "v3h_proto_opsd_instrdialogpp_seed456",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def setup_env() -> None:
    os.chdir(ROOT)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["HF_HOME"] = "/root/autodl-tmp/hf_cache"
    os.environ["TRANSFORMERS_CACHE"] = "/root/autodl-tmp/hf_cache"
    os.environ["HF_HUB_CACHE"] = "/root/autodl-tmp/hf_cache/hub"
    os.environ["TORCH_HOME"] = "/root/autodl-tmp/torch_cache"
    os.environ["PYTHONPATH"] = str(ROOT)
    os.environ["PYTHONUNBUFFERED"] = "1"
    LOGDIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


def load_summary() -> dict:
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {"runs": [], "status": "running", "started_at": now_iso(), "holdout_blocked": False}


def save_summary(data: dict) -> None:
    data["updated_at"] = now_iso()
    SUMMARY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def record(name: str, status: str, extra: str = "") -> None:
    line = f"{now_iso()}\t{name}\t{status}\t{extra}\n"
    with (LOGDIR / "acl_queue.steps.tsv").open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(f"{now_iso()} STEP {name} {status} {extra}", flush=True)


def gpu_busy() -> bool:
    ps = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True)
    for line in (ps.stdout or "").splitlines():
        if "core.trainer" in line or "core.train" in line:
            if "run_v3_acl_queue" in line or "grep" in line:
                continue
            if "python" in line:
                return True
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    text = (smi.stdout or "").strip()
    return bool(text)


def wait_gpu_idle() -> None:
    n = 0
    while gpu_busy():
        n += 1
        print(f"{now_iso()} GPU busy; sleep 30 [{n}]", flush=True)
        subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
            check=False,
        )
        time.sleep(30)


def nvidia_mem() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "").strip()


def config_path(name: str) -> Path:
    return ROOT / "configs" / f"{name}.yaml"


def launch_cmd(name: str) -> list:
    return [PYTHON, "-m", "core.trainer", "--config", str(config_path(name))]


def write_resume(name: str) -> None:
    cmd = " ".join(launch_cmd(name))
    text = (
        f"cd {ROOT}\n"
        f"export PYTHONPATH={ROOT}\n"
        f"export HF_HOME=/root/autodl-tmp/hf_cache\n"
        f"export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache\n"
        f"export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub\n"
        f"export TORCH_HOME=/root/autodl-tmp/torch_cache\n"
        f"export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        f"{cmd}\n"
    )
    (LOGDIR / f"{name}.resume_cmd").write_text(text, encoding="utf-8")


def existing_status(name: str, cfg: dict) -> str:
    run_dir = RESULTS / name
    if not run_dir.exists():
        return "absent"
    metrics = run_dir / "final_metrics.json"
    if not metrics.is_file():
        return "incomplete"
    saved = load_snapshot(run_dir / "config_snapshot.yaml")
    if saved and fingerprints_match(saved, cfg):
        return "match"
    return "mismatch"


def archive_incomplete(name: str, reason: str) -> Path | None:
    run_dir = RESULTS / name
    if not run_dir.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dest = RESULTS / f"{name}.{reason}.{stamp}"
    if dest.exists():
        dest = RESULTS / f"{name}.{reason}.{stamp}.{os.getpid()}"
    shutil.move(str(run_dir), str(dest))
    record(name, "ARCHIVED", str(dest))
    return dest


def run_one(name: str, summary: dict, *, oom_retry: bool = True) -> dict:
    cfg_file = config_path(name)
    if not cfg_file.is_file():
        raise FileNotFoundError(str(cfg_file))
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    write_resume(name)
    status = existing_status(name, cfg)
    if status == "match":
        rec = {
            "name": name,
            "ok": True,
            "skipped": True,
            "run_dir": str(RESULTS / name),
            "final_metrics": str(RESULTS / name / "final_metrics.json"),
        }
        summary["runs"].append(rec)
        save_summary(summary)
        record(name, "SKIP", "final_metrics matches config")
        return rec
    if status in {"incomplete", "mismatch"}:
        rec = {
            "name": name,
            "ok": False,
            "skipped": True,
            "blocked": status,
            "run_dir": str(RESULTS / name),
            "resume_cmd": str(LOGDIR / f"{name}.resume_cmd"),
        }
        summary["runs"].append(rec)
        save_summary(summary)
        record(name, "REFUSE", status)
        return rec

    wait_gpu_idle()
    if gpu_busy():
        raise RuntimeError(f"GPU occupied before {name}")

    log_path = LOGDIR / f"{name}.log"
    (LOGDIR / f"{name}.config.yaml").write_text(cfg_file.read_text(encoding="utf-8"), encoding="utf-8")
    (LOGDIR / f"{name}.start").write_text(now_iso() + "\n", encoding="utf-8")
    (LOGDIR / f"{name}.gpu_before").write_text(nvidia_mem() + "\n", encoding="utf-8")
    print(f"===== START {name} =====", flush=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(launch_cmd(name), cwd=str(ROOT), stdout=handle, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    (LOGDIR / f"{name}.end").write_text(now_iso() + "\n", encoding="utf-8")
    (LOGDIR / f"{name}.elapsed_sec").write_text(f"{elapsed:.1f}\n", encoding="utf-8")
    (LOGDIR / f"{name}.exitcode").write_text(str(proc.returncode) + "\n", encoding="utf-8")
    (LOGDIR / f"{name}.gpu_after").write_text(nvidia_mem() + "\n", encoding="utf-8")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    oom = "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text
    metrics = RESULTS / name / "final_metrics.json"
    if proc.returncode == 0 and metrics.is_file():
        rec = {
            "name": name,
            "ok": True,
            "skipped": False,
            "elapsed_sec": elapsed,
            "run_dir": str(RESULTS / name),
            "final_metrics": str(metrics),
            "gpu_after": nvidia_mem(),
        }
        summary["runs"].append(rec)
        save_summary(summary)
        record(name, "OK", f"sec={elapsed:.0f}")
        return rec
    if proc.returncode != 0 and oom and oom_retry:
        archived = archive_incomplete(name, "oom")
        record(name, "OOM_RETRY", f"archived={archived}")
        return run_one(name, summary, oom_retry=False)
    if proc.returncode in {0, 3, 4}:
        reason = explain_exit(proc.returncode)
    else:
        reason = f"exit={proc.returncode}"
    rec = {
        "name": name,
        "ok": False,
        "skipped": False,
        "elapsed_sec": elapsed,
        "exit_code": proc.returncode,
        "reason": reason,
        "oom": oom,
        "run_dir": str(RESULTS / name),
        "final_metrics": str(metrics) if metrics.is_file() else "",
        "resume_cmd": str(LOGDIR / f"{name}.resume_cmd"),
        "log": str(log_path),
    }
    summary["runs"].append(rec)
    save_summary(summary)
    record(name, "FAIL", reason)
    return rec


def smoke_audit(summary: dict) -> bool:
    checker = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / "check_v3h_smoke.py"), SMOKE],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    text = (checker.stdout or "") + (checker.stderr or "")
    (LOGDIR / "smoke_audit.out").write_text(text, encoding="utf-8")
    ok = checker.returncode == 0 and "SMOKE_HOLDOUT_OK" in text
    summary["smoke_audit_ok"] = ok
    summary["smoke_audit_exit"] = checker.returncode
    if not ok:
        summary["holdout_blocked"] = True
        fail_path = RUNS_ROOT / "smoke_leakage_failure.json"
        fail_path.write_text(
            json.dumps(
                {
                    "run": SMOKE,
                    "exit": checker.returncode,
                    "output": text,
                    "holdout_queue": "blocked",
                    "old_protocol_12": "allowed",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        record(SMOKE, "AUDIT_FAIL", text[-500:].replace("\n", " | "))
    else:
        record(SMOKE, "AUDIT_OK", "")
    save_summary(summary)
    print(text, flush=True)
    return ok


def run_names(names: list, summary: dict) -> None:
    for name in names:
        run_one(name, summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "old12", "holdout8", "all"), default="all")
    args = parser.parse_args()
    setup_env()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another v3 queue holds /tmp/rp_lora_v3_gpu.lock", flush=True)
        raise SystemExit(2)
    summary = load_summary()
    summary["status"] = "running"
    save_summary(summary)
    try:
        if args.mode in {"smoke", "all"}:
            smoke_rec = run_one(SMOKE, summary)
            if not smoke_rec.get("ok"):
                summary["holdout_blocked"] = True
                summary["smoke_failed"] = True
                save_summary(summary)
                record(SMOKE, "SMOKE_NOT_OK", "holdout queue blocked; old 12 still allowed")
            else:
                smoke_audit(summary)
        if args.mode in {"old12", "all"}:
            run_names(OLD_12, summary)
        if args.mode in {"holdout8", "all"}:
            summary = load_summary()
            if summary.get("holdout_blocked"):
                record("v3h", "BLOCKED", "smoke leakage or smoke failure")
            else:
                run_names(HOLDOUT_8, summary)
        summary = load_summary()
        failed = [row["name"] for row in summary["runs"] if not row.get("ok")]
        summary["status"] = "failed" if failed else "ok"
        summary["failed"] = failed
        save_summary(summary)
        print(json.dumps({"status": summary["status"], "failed": failed}, indent=2), flush=True)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
