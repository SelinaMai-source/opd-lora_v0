#!/usr/bin/env python3
"""Serial Protocol B 3-seed queue. One trainer. Never overwrites a run directory.

Order is Phase A (8 PGDR) then Phase B (8 random). holdout_seed stays 456.
Reuses lock / fingerprint skip / refuse-overwrite from run_v3_acl_queue.py.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/root/research-repos/opd-lora_v3")
sys.path.insert(0, str(ROOT / "scripts"))
import run_v3_acl_queue as acl  # noqa: E402

acl.SUMMARY = acl.LOGDIR / "p0_3seed_queue_summary.json"
QUEUE_DOC = ROOT / "docs" / "P0_P1_Remaining_Run_Queue.md"
MANIFEST = ROOT / "results" / "manifest.csv"
DASH = "—"

VERIFIED_456 = [
    "v3h_rand_ce_instrdialog_seed456",
    "v3h_proto_ce_instrdialog_seed456",
    "v3h_rand_opsd_instrdialog_seed456",
    "v3h_proto_opsd_instrdialog_seed456",
    "v3h_rand_ce_instrdialogpp_seed456",
    "v3h_proto_ce_instrdialogpp_seed456",
    "v3h_rand_opsd_instrdialogpp_seed456",
    "v3h_proto_opsd_instrdialogpp_seed456",
]
PHASE_A = [
    "v3h_proto_ce_instrdialog_seed789",
    "v3h_proto_ce_instrdialog_seed42",
    "v3h_proto_opsd_instrdialog_seed789",
    "v3h_proto_opsd_instrdialog_seed42",
    "v3h_proto_ce_instrdialogpp_seed789",
    "v3h_proto_ce_instrdialogpp_seed42",
    "v3h_proto_opsd_instrdialogpp_seed789",
    "v3h_proto_opsd_instrdialogpp_seed42",
]
PHASE_B = [
    "v3h_rand_ce_instrdialog_seed789",
    "v3h_rand_ce_instrdialog_seed42",
    "v3h_rand_opsd_instrdialog_seed789",
    "v3h_rand_opsd_instrdialog_seed42",
    "v3h_rand_ce_instrdialogpp_seed789",
    "v3h_rand_ce_instrdialogpp_seed42",
    "v3h_rand_opsd_instrdialogpp_seed789",
    "v3h_rand_opsd_instrdialogpp_seed42",
]
QUEUED_16 = PHASE_A + PHASE_B

GPU_456 = {
    "v3h_rand_ce_instrdialog_seed456": "624s",
    "v3h_proto_ce_instrdialog_seed456": "644s",
    "v3h_rand_opsd_instrdialog_seed456": "1257s",
    "v3h_proto_opsd_instrdialog_seed456": "1246s",
    "v3h_rand_ce_instrdialogpp_seed456": "1894s",
    "v3h_proto_ce_instrdialogpp_seed456": "1829s",
    "v3h_rand_opsd_instrdialogpp_seed456": "3110s",
    "v3h_proto_opsd_instrdialogpp_seed456": "2951s",
}

HEADER = (
    "| Run ID | Benchmark | Method | Seed | Protocol | Status | Config | Result Path | GPU Time | Notes |"
)
SEP = "|---|---|---|---|---|---|---|---|---|---|"


def parse_name(name: str) -> dict:
    parts = name.split("_")
    # v3h_{proto|rand}_{ce|opsd}_{instrdialog|instrdialogpp}_seed{N}
    method = f"{parts[1]}_{parts[2]}"
    if "instrdialogpp" in name:
        bench = "instrdialogpp"
    else:
        bench = "instrdialog"
    seed = int(name.rsplit("seed", 1)[1])
    return {"method": method, "benchmark": bench, "seed": seed}


def live_status(name: str) -> tuple[str, str, str]:
    """Return (status, result_path, gpu_time)."""
    run_dir = acl.RESULTS / name
    metrics = run_dir / "final_metrics.json"
    elapsed = acl.LOGDIR / f"{name}.elapsed_sec"
    gpu = DASH
    if elapsed.is_file():
        gpu = f"{elapsed.read_text(encoding='utf-8').strip()}s"
    elif name in GPU_456:
        gpu = GPU_456[name]
    if metrics.is_file():
        if name in VERIFIED_456:
            return "Verified Completed", str(run_dir), gpu
        return "Completed", str(run_dir), gpu
    start = acl.LOGDIR / f"{name}.start"
    end = acl.LOGDIR / f"{name}.end"
    if start.is_file() and not end.is_file() and run_dir.exists():
        return "Running", str(run_dir), gpu
    if run_dir.exists() and not metrics.is_file():
        return "Incomplete", str(run_dir), gpu
    return "Queued", DASH, DASH


def notes_for(name: str, status: str) -> str:
    info = parse_name(name)
    proto = "PGDR 每段约 5 条监控前向；proto 回放 8（4+4）" if info["method"].startswith("proto") else "uniform 回放 9；无监控前向选样"
    if name in VERIFIED_456:
        return f"复用，不重跑；{proto}；holdout_seed=456"
    if status == "Completed":
        return f"新跑完；{proto}；holdout_seed=456"
    if status == "Running":
        return f"进行中；{proto}；holdout_seed=456"
    if status == "Incomplete":
        resume = acl.LOGDIR / f"{name}.resume_cmd"
        extra = f"；续跑见 {resume}" if resume.is_file() else ""
        return f"无 final_metrics.json，目录保留{extra}；{proto}"
    if status == "Failed":
        resume = acl.LOGDIR / f"{name}.resume_cmd"
        return f"失败，目录保留；续跑见 {resume}；{proto}"
    return f"{proto}；holdout_seed 必须 456"


def queue_row(name: str, status_override: str | None = None) -> str:
    info = parse_name(name)
    status, result, gpu = live_status(name)
    if status_override:
        status = status_override
    cfg = f"configs/{name}.yaml"
    return (
        f"| {name} | {info['benchmark']} | {info['method']} | {info['seed']} | "
        f"Protocol B train_holdout | {status} | {cfg} | {result} | {gpu} | {notes_for(name, status)} |"
    )


def write_queue_doc(*, current: str | None = None, extra: str = "") -> None:
    lines = [
        "# P0 / P1 Remaining Run Queue",
        "",
        "Protocol B 三种子 2×2。8 个 seed 456 已 Verified Completed，不重跑。",
        "16 个新 run 的 `holdout_seed` 固定 456，只改训练 `seed` 与 `output.run_name`。",
        "训练入口：`/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python -m core.trainer --config <yaml>`。",
        "结果根：`/root/autodl-tmp/rp_lora_v3_runs`。单进程，锁 `/tmp/rp_lora_v3_gpu.lock`。",
        "指纹一致则跳过；目录已在且不完整则拒绝覆盖。不启动 E2–E12。",
        "",
        HEADER,
        SEP,
    ]
    for name in VERIFIED_456:
        lines.append(queue_row(name, "Verified Completed"))
    for name in QUEUED_16:
        override = None
        if current and name == current:
            override = "Running"
        lines.append(queue_row(name, override))
    lines.extend(
        [
            "",
            "## Resume",
            "",
            "中断后不要覆盖已有目录。原样续跑整队：",
            "",
            "```bash",
            f"cd {ROOT}",
            "export PYTHONPATH=/root/research-repos/opd-lora_v3",
            "export HF_HOME=/root/autodl-tmp/hf_cache",
            "export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache",
            "export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub",
            "export TORCH_HOME=/root/autodl-tmp/torch_cache",
            "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python scripts/run_v3h_p0_3seed_queue.py",
            "```",
            "",
            "单条续跑命令在 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。",
        ]
    )
    if extra:
        lines.extend(["", extra])
    QUEUE_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(x) for x in value)
    return str(value)


def upsert_manifest(name: str) -> None:
    run_dir = acl.RESULTS / name
    metrics_path = run_dir / "final_metrics.json"
    if not metrics_path.is_file():
        return
    cfg_path = run_dir / "config_snapshot.yaml"
    cfg = acl.load_snapshot(cfg_path)
    replay = cfg.get("replay") if isinstance(cfg.get("replay"), dict) else {}
    ce = cfg.get("ce_opsd") if isinstance(cfg.get("ce_opsd"), dict) else {}
    lora = cfg.get("lora") if isinstance(cfg.get("lora"), dict) else {}
    train = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    final = payload.get("final") or {}
    seen = final.get("eval.num_seen_segments")
    jsonl = run_dir / "metrics.jsonl"
    steps = None
    tokens = None
    if jsonl.is_file():
        last = {}
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get("segment_id")
            last[sid] = row
        total_steps = 0.0
        total_tokens = 0.0
        have_steps = False
        have_tokens = False
        for row in last.values():
            for key in ("train.batches", "train.train.n_steps"):
                if isinstance(row.get(key), (int, float)):
                    total_steps += float(row[key])
                    have_steps = True
                    break
            for key in ("train.num_supervised_tokens", "train.num_total_tokens"):
                if isinstance(row.get(key), (int, float)):
                    total_tokens += float(row[key])
                    have_tokens = True
                    break
        if have_steps:
            steps = int(total_steps)
        if have_tokens:
            tokens = int(total_tokens)
    info = parse_name(name)
    strategy = replay.get("strategy")
    lam = ce.get("lambda_opsd")
    method = f"ce_opsd_replay|strategy={strategy}|lambda={lam}|slot={replay.get('transfer_slot', 'neighbor')}"
    if strategy == "uniform":
        method += "|coverage_dedup=False" if replay.get("coverage_dedup") is False else ""
        ver = f"measured;num_seen_segments={seen};monitor=train_holdout;uniform_replay_does_not_call_select_forgotten"
    else:
        ver = f"measured;num_seen_segments={seen};monitor=train_holdout"
    stream = paths.get("processed_stream_file")
    proto = (
        f"format_normalize=true;monitor=train_holdout;max_segments={data.get('max_segments', -1)};stream={stream}"
    )
    ckpt = run_dir / "final_adapter" / "adapter_model.safetensors"
    pred = run_dir / "eval_debug"
    row = {
        "run_id": name,
        "experiment_version": "v3h",
        "benchmark": info["benchmark"],
        "method": method,
        "seed": info["seed"],
        "model": model.get("hf_model_name_or_path"),
        "lora_rank": lora.get("r"),
        "lora_alpha": lora.get("alpha"),
        "target_modules": _csv_cell(lora.get("target_modules")),
        "learning_rate": train.get("lr"),
        "memory_capacity": replay.get("buffer_k", 10),
        "replay_ratio": replay.get("replay_ratio"),
        "related_ratio": replay.get("neighbor_ratio"),
        "degraded_ratio": replay.get("forgotten_ratio"),
        "opsd_lambda": lam,
        "training_steps": steps if steps is not None else "",
        "training_tokens": tokens if tokens is not None else "",
        "evaluation_protocol": proto,
        "config_path": str(cfg_path) if cfg_path.is_file() else "",
        "source_commit": "",
        "checkpoint_path": str(ckpt) if ckpt.is_file() else "",
        "metrics_path": str(metrics_path),
        "prediction_path": str(pred) if pred.is_dir() else "",
        "run_status": "completed",
        "verification_status": ver,
    }
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or list(row.keys())
        existing = list(reader)
    replaced = False
    for i, old in enumerate(existing):
        if old.get("run_id") == name:
            existing[i] = {key: _csv_cell(row.get(key, old.get(key, ""))) for key in fields}
            replaced = True
            break
    if not replaced:
        existing.append({key: _csv_cell(row.get(key, "")) for key in fields})
    existing.sort(key=lambda item: item.get("run_id") or "")
    with MANIFEST.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)


def verify_holdout_seed(name: str) -> None:
    cfg = acl.yaml.safe_load(acl.config_path(name).read_text(encoding="utf-8"))
    replay = cfg.get("replay") or {}
    if int(replay.get("holdout_seed", -1)) != 456:
        raise SystemExit(f"{name} holdout_seed={replay.get('holdout_seed')} (must be 456)")
    if str(replay.get("monitor")) != "train_holdout":
        raise SystemExit(f"{name} monitor={replay.get('monitor')} (must be train_holdout)")
    if int(cfg.get("seed", -1)) not in {789, 42}:
        raise SystemExit(f"{name} seed={cfg.get('seed')} (must be 789 or 42)")
    if str((cfg.get("output") or {}).get("run_name")) != name:
        raise SystemExit(f"{name} run_name mismatch")


def main() -> None:
    import fcntl

    acl.setup_env()
    acl.LOGDIR.mkdir(parents=True, exist_ok=True)
    for name in QUEUED_16:
        verify_holdout_seed(name)
    write_queue_doc()
    lock_handle = acl.LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another v3 queue holds /tmp/rp_lora_v3_gpu.lock", flush=True)
        raise SystemExit(2)
    summary = {"runs": [], "status": "running", "started_at": acl.now_iso(), "queue": "p0_3seed"}
    acl.save_summary(summary)
    try:
        for name in QUEUED_16:
            write_queue_doc(current=name)
            rec = acl.run_one(name, summary)
            if rec.get("ok") and (acl.RESULTS / name / "final_metrics.json").is_file():
                upsert_manifest(name)
            write_queue_doc()
        failed = [row["name"] for row in summary["runs"] if not row.get("ok")]
        summary["status"] = "failed" if failed else "ok"
        summary["failed"] = failed
        acl.save_summary(summary)
        write_queue_doc(extra=f"queue_status={summary['status']} failed={failed}")
        print(json.dumps({"status": summary["status"], "failed": failed}, indent=2), flush=True)
        raise SystemExit(1 if failed else 0)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
