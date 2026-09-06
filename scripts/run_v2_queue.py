#!/usr/bin/env python3
"""Serial v2 seed-456 queue. Single RTX 4090. Never start a second 8B job.

Phases:
  --mode smoke  ce_opsd 2-seg + proto_replay 2-seg
  --mode full   20 formal jobs (skip if run_name/final_metrics.json exists)
  --mode all    smoke then full (stop if either smoke lacks final_metrics.json)

Does not rerun CITB / LCIA / v0 / v1.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/rp_lora_v0")
PYTHON = "/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python"
LOGDIR = ROOT / "logs"
RESULTS = ROOT / "results" / "runs"
SUMMARY = LOGDIR / "v2_queue_summary.json"
LOCK = Path("/tmp/rp_lora_v0_gpu.lock")
STEPS = LOGDIR / "v2_queue.steps.tsv"
SEED = 456

SMOKE_JOBS = [
    {
        "name": "v2_multiref_gold_opsd_instrdialog_l02_seed456_smoke2seg",
        "config": "configs/v2_multiref_gold_opsd_instrdialog_l02_seed456_smoke2seg.yaml",
        "run_name": "v2_multiref_gold_opsd_instrdialog_l02_seed456_smoke2seg",
    },
    {
        "name": "v2_proto_replay_opsd_instrdialog_seed456_smoke2seg",
        "config": "configs/v2_proto_replay_opsd_instrdialog_seed456_smoke2seg.yaml",
        "run_name": "v2_proto_replay_opsd_instrdialog_seed456_smoke2seg",
    },
]

FORMAL_JOBS = [
    {"name": "v2_multiref_gold_opsd_instrdialog_l01_seed456", "config": "configs/v2_multiref_gold_opsd_instrdialog_l01_seed456.yaml"},
    {"name": "v2_multiref_gold_opsd_instrdialog_l02_seed456", "config": "configs/v2_multiref_gold_opsd_instrdialog_l02_seed456.yaml"},
    {"name": "v2_multiref_gold_opsd_instrdialog_l03_seed456", "config": "configs/v2_multiref_gold_opsd_instrdialog_l03_seed456.yaml"},
    {"name": "v2_multiref_gold_opsd_instrdialogpp_l01_seed456", "config": "configs/v2_multiref_gold_opsd_instrdialogpp_l01_seed456.yaml"},
    {"name": "v2_multiref_gold_opsd_instrdialogpp_l02_seed456", "config": "configs/v2_multiref_gold_opsd_instrdialogpp_l02_seed456.yaml"},
    {"name": "v2_multiref_gold_opsd_instrdialogpp_l03_seed456", "config": "configs/v2_multiref_gold_opsd_instrdialogpp_l03_seed456.yaml"},
    {"name": "v2_rand_replay_opsd_instrdialog_seed456", "config": "configs/v2_rand_replay_opsd_instrdialog_seed456.yaml"},
    {"name": "v2_rand_replay_opsd_instrdialogpp_seed456", "config": "configs/v2_rand_replay_opsd_instrdialogpp_seed456.yaml"},
    {"name": "v2_proto_replay_opsd_instrdialog_seed456", "config": "configs/v2_proto_replay_opsd_instrdialog_seed456.yaml"},
    {"name": "v2_proto_replay_opsd_instrdialogpp_seed456", "config": "configs/v2_proto_replay_opsd_instrdialogpp_seed456.yaml"},
    {"name": "v2_proto_teacher_opsd_random_instrdialog_seed456", "config": "configs/v2_proto_teacher_opsd_random_instrdialog_seed456.yaml"},
    {"name": "v2_proto_teacher_opsd_random_instrdialogpp_seed456", "config": "configs/v2_proto_teacher_opsd_random_instrdialogpp_seed456.yaml"},
    {"name": "v2_proto_teacher_opsd_matched_instrdialog_seed456", "config": "configs/v2_proto_teacher_opsd_matched_instrdialog_seed456.yaml"},
    {"name": "v2_proto_teacher_opsd_matched_instrdialogpp_seed456", "config": "configs/v2_proto_teacher_opsd_matched_instrdialogpp_seed456.yaml"},
    {"name": "v2_metric_opsd_instrdialog_m005_seed456", "config": "configs/v2_metric_opsd_instrdialog_m005_seed456.yaml"},
    {"name": "v2_metric_opsd_instrdialog_m01_seed456", "config": "configs/v2_metric_opsd_instrdialog_m01_seed456.yaml"},
    {"name": "v2_metric_opsd_instrdialog_m02_seed456", "config": "configs/v2_metric_opsd_instrdialog_m02_seed456.yaml"},
    {"name": "v2_metric_opsd_instrdialogpp_m005_seed456", "config": "configs/v2_metric_opsd_instrdialogpp_m005_seed456.yaml"},
    {"name": "v2_metric_opsd_instrdialogpp_m01_seed456", "config": "configs/v2_metric_opsd_instrdialogpp_m01_seed456.yaml"},
    {"name": "v2_metric_opsd_instrdialogpp_m02_seed456", "config": "configs/v2_metric_opsd_instrdialogpp_m02_seed456.yaml"},
]
for job in FORMAL_JOBS:
    job.setdefault("run_name", job["name"])


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
    LOGDIR.mkdir(parents=True, exist_ok=True)


def load_summary() -> dict:
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {"runs": [], "status": "running", "started_at": now_iso(), "seed": SEED}


def save_summary(data: dict) -> None:
    data["updated_at"] = now_iso()
    SUMMARY.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def record_step(name: str, status: str, extra: str = "") -> None:
    line = f"{now_iso()}\t{name}\t{status}\t{extra}\n"
    with STEPS.open("a", encoding="utf-8") as f:
        f.write(line)
    print(f"{now_iso()} STEP {name} {status} {extra}", flush=True)


def gpu_busy() -> bool:
    ps = subprocess.run(["ps", "-eo", "pid,cmd"], capture_output=True, text=True)
    for line in (ps.stdout or "").splitlines():
        if " -m core.train" in line and "grep" not in line:
            return True
    p = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return any(ch.isdigit() for ch in (p.stdout or ""))


def wait_gpu_idle() -> None:
    n = 0
    while gpu_busy():
        n += 1
        print(f"{now_iso()} GPU busy; sleep 30 [{n}]", flush=True)
        subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]
        )
        time.sleep(30)


def extract_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "final_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(str(metrics_path))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    final = metrics.get("final") or {}
    return {
        "run_dir": str(run_dir),
        "run_id": metrics.get("run_id") or run_dir.name,
        "seen_avg_score": final.get("eval.seen_avg_score"),
        "forgetting": final.get("eval.forgetting"),
        "token_f1_mean": final.get("eval.token_f1_mean"),
        "final_metrics": str(metrics_path),
    }


def job_done(run_name: str) -> Path | None:
    run_dir = RESULTS / run_name
    if (run_dir / "final_metrics.json").is_file():
        return run_dir
    return None


def run_one(name: str, config: Path, run_name: str, summary: dict, oom_retry: bool = True) -> dict:
    existing = job_done(run_name)
    if existing is not None:
        mets = extract_metrics(existing)
        rec = {"name": name, "config": str(config), "ok": True, "skipped": True, **mets}
        if not any(r.get("name") == name and r.get("ok") for r in summary["runs"]):
            summary["runs"].append(rec)
            save_summary(summary)
        record_step(name, "SKIP", f"already_ok run={existing}")
        print(f"===== SKIP {name} {existing} =====", flush=True)
        return rec

    if not config.is_file():
        raise FileNotFoundError(str(config))

    wait_gpu_idle()
    if gpu_busy():
        raise RuntimeError(f"GPU occupied before {name}")

    log_path = LOGDIR / f"{name}.log"
    start_path = LOGDIR / f"{name}.start"
    end_path = LOGDIR / f"{name}.end"
    exit_path = LOGDIR / f"{name}.exitcode"

    print(f"===== START {name} config={config} =====", flush=True)
    start_path.write_text(now_iso() + "\n", encoding="utf-8")
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [PYTHON, "-m", "core.train", "--config", str(config)],
            cwd=str(ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    exit_path.write_text(str(proc.returncode) + "\n", encoding="utf-8")
    end_path.write_text(now_iso() + "\n", encoding="utf-8")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    oom = "OutOfMemoryError" in log_text or "CUDA out of memory" in log_text

    if proc.returncode != 0 and oom and oom_retry:
        print(f"OOM on {name}; retry once", flush=True)
        record_step(name, "OOM_RETRY", f"sec={elapsed:.0f}")
        wait_gpu_idle()
        return run_one(name, config, run_name, summary, oom_retry=False)

    if proc.returncode != 0:
        rec = {
            "name": name,
            "config": str(config),
            "ok": False,
            "exit_code": proc.returncode,
            "seconds": elapsed,
            "oom": oom,
            "log": str(log_path),
        }
        summary["runs"].append(rec)
        summary["status"] = f"failed:{name}"
        save_summary(summary)
        record_step(name, "FAIL", f"exit={proc.returncode}")
        print(f"FAIL {name} exit={proc.returncode}", flush=True)
        print(log_text[-4000:], flush=True)
        raise SystemExit(proc.returncode)

    run_dir = RESULTS / run_name
    if not (run_dir / "final_metrics.json").is_file():
        summary["status"] = f"failed:{name}:no_metrics"
        save_summary(summary)
        record_step(name, "FAIL", "missing final_metrics.json")
        raise SystemExit(f"missing final_metrics.json for {run_name}")

    mets = extract_metrics(run_dir)
    rec = {
        "name": name,
        "config": str(config),
        "ok": True,
        "skipped": False,
        "exit_code": 0,
        "seconds": elapsed,
        "duration_hms": time.strftime("%H:%M:%S", time.gmtime(elapsed)),
        **mets,
    }
    summary["runs"].append(rec)
    save_summary(summary)
    record_step(name, "PASS", f"seen={mets['seen_avg_score']} forget={mets['forgetting']} sec={elapsed:.0f}")
    print(
        f"===== PASS {name} dir={mets['run_dir']} "
        f"seen={mets['seen_avg_score']} forget={mets['forgetting']} "
        f"sec={elapsed:.0f} =====",
        flush=True,
    )
    return rec


def acquire_lock():
    lock_f = open(LOCK, "w")
    while True:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            print(f"{now_iso()} acquired {LOCK}", flush=True)
            return lock_f
        except BlockingIOError:
            print(f"{now_iso()} LOCK_BUSY {LOCK}; sleep 30", flush=True)
            time.sleep(30)


def run_jobs(jobs: list[dict], summary: dict, label: str) -> None:
    for i, job in enumerate(jobs, 1):
        print(f"----- {label} {i}/{len(jobs)} {job['name']} -----", flush=True)
        run_one(job["name"], ROOT / job["config"], job.get("run_name", job["name"]), summary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full", "all"], default="all")
    args = parser.parse_args()
    setup_env()

    lock_f = acquire_lock()
    try:
        summary = load_summary()
        summary["status"] = f"running:{args.mode}"
        save_summary(summary)
        if args.mode in {"smoke", "all"}:
            run_jobs(SMOKE_JOBS, summary, "smoke")
            for job in SMOKE_JOBS:
                if job_done(job["run_name"]) is None:
                    summary["status"] = f"failed:smoke:{job['run_name']}"
                    save_summary(summary)
                    raise SystemExit(f"smoke missing final_metrics.json: {job['run_name']}")
            summary["smoke_ok"] = True
            save_summary(summary)
        if args.mode in {"full", "all"}:
            if args.mode == "full":
                missing = [j["run_name"] for j in SMOKE_JOBS if job_done(j["run_name"]) is None]
                if missing:
                    raise SystemExit(f"refusing full: smoke not passed: {missing}")
            run_jobs(FORMAL_JOBS, summary, "formal")
        summary["status"] = "completed"
        save_summary(summary)
        print("ALL_V2_QUEUE_COMPLETED", flush=True)
        return 0
    finally:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_f.close()


if __name__ == "__main__":
    raise SystemExit(main())
