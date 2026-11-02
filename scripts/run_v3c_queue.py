#!/usr/bin/env python3
"""Serial v3c queue. One trainer. Refuses a config that is not the corrected protocol.

Phase 1: seed 456, R/U/P x CE/CE+OPSD x two streams (12).
Phase 2: R and P x two objectives x two streams x seeds 789 and 42 (16).
Phase 3: U x two objectives x two streams x seeds 789 and 42 (8).
Seed 456 R/P from phase 1 are not repeated. A matching fingerprint is a skip.
An existing directory is not overwritten and is not marked completed without
final_metrics.json.
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
SUMMARY = LOGDIR / "v3c_queue_summary.json"
LOCK = Path("/tmp/rp_lora_v3_gpu.lock")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from core.replay_budget import validate_v3c_replay_block  # noqa: E402
from core.run_guard import explain_exit, fingerprints_match, load_snapshot  # noqa: E402
from v3c_matrix import all_runs, phase1, phase2, phase3  # noqa: E402


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
    return {"runs": [], "status": "running", "started_at": now_iso()}


def save_summary(data: dict) -> None:
    data["updated_at"] = now_iso()
    SUMMARY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def record(name: str, status: str, extra: str = "") -> None:
    line = f"{now_iso()}\t{name}\t{status}\t{extra}\n"
    with (LOGDIR / "v3c_queue.steps.tsv").open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(f"{now_iso()} STEP {name} {status} {extra}", flush=True)


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
        f"# Incomplete directories are kept. core.trainer refuses to overwrite them.\n"
        f"{cmd}\n"
        f"# Full queue, skipping runs that already have a matching final_metrics.json:\n"
        f"{PYTHON} {ROOT / 'scripts' / 'run_v3c_queue.py'}\n"
    )
    (LOGDIR / f"{name}.resume_cmd").write_text(text, encoding="utf-8")
    (LOGDIR / "v3c_queue.resume_cmd").write_text(
        f"cd {ROOT}\nexport PYTHONPATH={ROOT}\n{PYTHON} {ROOT / 'scripts' / 'run_v3c_queue.py'}\n",
        encoding="utf-8",
    )


def gpu_busy() -> bool:
    ps = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True)
    for line in (ps.stdout or "").splitlines():
        if "core.trainer" in line or "core.train" in line:
            if "run_v3c_queue" in line or "grep" in line:
                continue
            if "python" in line:
                return True
    smi = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return bool((smi.stdout or "").strip())


def wait_gpu_idle() -> None:
    n = 0
    while gpu_busy():
        n += 1
        print(f"{now_iso()} GPU busy; sleep 30 [{n}]", flush=True)
        time.sleep(30)
        if n > 40:
            raise RuntimeError("GPU stayed busy")


def nvidia_mem() -> str:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return (proc.stdout or "").strip()


def load_cfg(name: str) -> dict:
    path = config_path(name)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} is not a mapping")
    if not name.startswith("v3c_"):
        raise ValueError(f"refusing non-v3c run {name}")
    if "v3h_" in name:
        raise ValueError(f"refusing to touch a v3h name: {name}")
    replay = cfg.get("replay") if isinstance(cfg.get("replay"), dict) else {}
    validate_v3c_replay_block(replay)
    output = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    if str(output.get("run_name")) != name:
        raise ValueError(f"{name} run_name is {output.get('run_name')}")
    norm = cfg.get("eval_normalization") if isinstance(cfg.get("eval_normalization"), dict) else {}
    if norm.get("format_normalize") is not True:
        raise ValueError(f"{name} format_normalize is not true")
    return cfg


def preflight(names: list) -> None:
    if not names:
        raise SystemExit("empty v3c queue")
    for name in names:
        load_cfg(name)
    print(f"{now_iso()} PREFLIGHT ok {len(names)} configs", flush=True)


def existing_status(name: str, cfg: dict) -> str:
    run_dir = RESULTS / name
    if name.startswith("v3h_") or not name.startswith("v3c_"):
        raise RuntimeError(name)
    if not run_dir.exists():
        return "absent"
    metrics = run_dir / "final_metrics.json"
    if not metrics.is_file():
        return "incomplete"
    saved = load_snapshot(run_dir / "config_snapshot.yaml")
    if saved and fingerprints_match(saved, cfg):
        return "match"
    return "mismatch"


def archive_oom(name: str) -> Path | None:
    run_dir = RESULTS / name
    if not run_dir.exists():
        return None
    if not name.startswith("v3c_"):
        raise RuntimeError(f"refusing to move {name}")
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    dest = RESULTS / f"{name}.oom.{stamp}"
    shutil.move(str(run_dir), str(dest))
    record(name, "ARCHIVED_OOM", str(dest))
    return dest


def run_one(name: str, summary: dict, *, oom_retry: bool = True) -> dict:
    cfg = load_cfg(name)
    write_resume(name)
    status = existing_status(name, cfg)
    if status == "match":
        rec = {
            "name": name,
            "ok": True,
            "skipped": True,
            "completed": True,
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
            "completed": False,
            "blocked": status,
            "run_dir": str(RESULTS / name),
            "resume_cmd": str(LOGDIR / f"{name}.resume_cmd"),
        }
        summary["runs"].append(rec)
        save_summary(summary)
        record(name, "REFUSE", status)
        return rec

    wait_gpu_idle()
    log_path = LOGDIR / f"{name}.log"
    (LOGDIR / f"{name}.config.yaml").write_text(config_path(name).read_text(encoding="utf-8"), encoding="utf-8")
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
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    oom = "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text
    metrics = RESULTS / name / "final_metrics.json"
    if proc.returncode == 0 and metrics.is_file():
        rec = {
            "name": name,
            "ok": True,
            "skipped": False,
            "completed": True,
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
        archive_oom(name)
        record(name, "OOM_RETRY", "")
        return run_one(name, summary, oom_retry=False)
    reason = explain_exit(proc.returncode) if proc.returncode in {0, 3, 4} else f"exit={proc.returncode}"
    if proc.returncode == 0 and not metrics.is_file():
        reason = "exit0_without_final_metrics"
    rec = {
        "name": name,
        "ok": False,
        "skipped": False,
        "completed": False,
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


def selected_names(mode: str) -> list:
    if mode == "phase1":
        return phase1()
    if mode == "phase2":
        return phase2()
    if mode == "phase3":
        return phase3()
    if mode == "all":
        return all_runs()
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("phase1", "phase2", "phase3", "all"), default="all")
    args = parser.parse_args()
    setup_env()
    names = selected_names(args.phase)
    preflight(names)
    lock_handle = LOCK.open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another v3 queue holds /tmp/rp_lora_v3_gpu.lock", flush=True)
        raise SystemExit(2)
    summary = load_summary()
    summary["status"] = "running"
    summary["phase"] = args.phase
    save_summary(summary)
    stopped = False
    try:
        for name in names:
            rec = run_one(name, summary)
            if not rec.get("ok"):
                stopped = True
                summary["status"] = "stopped"
                summary["stopped_at"] = name
                save_summary(summary)
                print(f"V3C_QUEUE_STOP {name}", flush=True)
                break
        summary = load_summary()
        latest = {}
        for row in summary["runs"]:
            latest[row["name"]] = row
        finished = [name for name, row in latest.items() if row.get("completed")]
        failed = [name for name, row in latest.items() if not row.get("ok")]
        gpu_sec = sum(float(row.get("elapsed_sec") or 0.0) for row in summary["runs"])
        summary["completed"] = finished
        summary["failed"] = failed
        summary["gpu_seconds"] = gpu_sec
        if not stopped:
            summary["status"] = "ok" if not failed else "failed"
        save_summary(summary)
        tag = "V3C_QUEUE_DONE" if summary["status"] == "ok" else "V3C_QUEUE_FAIL"
        print(
            json.dumps(
                {
                    "tag": tag,
                    "status": summary["status"],
                    "completed": len(finished),
                    "failed": failed,
                    "gpu_seconds": gpu_sec,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if summary["status"] != "ok":
            raise SystemExit(1)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
