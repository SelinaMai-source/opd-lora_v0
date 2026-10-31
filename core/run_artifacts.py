from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def git_cmd(repo_dir: Path, args: List[str]) -> str:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(repo_dir), stderr=subprocess.DEVNULL).decode("utf-8").strip()
        return out
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def init_overfit_run_manifest(
    *,
    cfg: Dict[str, Any],
    run_id: str,
    config_path: str,
    config_snapshot_path: str,
    run_dir: Path,
    debug_tools: Dict[str, Any],
) -> Dict[str, Any]:
    dataset_path = ""
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    if str(cfg.get("mode", "")).strip() == "debug":
        dataset_path = str(paths_cfg.get("sample_stream_path", ""))
    else:
        dataset_path = str(paths_cfg.get("processed_stream_dir", ""))
    dataset_sha = ""
    p = Path(dataset_path) if dataset_path else None
    if p and p.is_file():
        dataset_sha = sha256_file(p)
    repo_dir = Path(__file__).resolve().parents[1]
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    tokenizer_name = str(model_cfg.get("hf_model_name_or_path", ""))
    return {
        "run_id": str(run_id),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "git_branch_name": git_cmd(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit_sha": git_cmd(repo_dir, ["rev-parse", "HEAD"]),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": str(dataset_sha),
        "tokenizer_path_or_name": tokenizer_name,
        "model_path_or_name": str(model_cfg.get("hf_model_name_or_path", "")),
        "lora_config": cfg.get("lora", {}),
        "decode_config": {
            "gen_max_new_tokens": model_cfg.get("gen_max_new_tokens"),
            "gen_do_sample": model_cfg.get("gen_do_sample"),
            "gen_num_beams": model_cfg.get("gen_num_beams"),
            "overfit_gen_max_new_tokens": debug_tools.get("overfit_gen_max_new_tokens"),
        },
        "enabled_diagnosis_modules": {
            "enable_sequence_behavior_diagnosis": bool(debug_tools.get("enable_sequence_behavior_diagnosis", False)),
            "run_failure_decomposition": bool(debug_tools.get("run_failure_decomposition", True)),
            "run_first_token_audit": bool(debug_tools.get("run_first_token_audit", True)),
            "run_prefix_rollout": bool(debug_tools.get("run_prefix_rollout", False)),
            "run_decode_ablation": bool(debug_tools.get("run_decode_ablation", False)),
            "run_overfit_ladder": bool(debug_tools.get("run_overfit_ladder", False)),
        },
        "artifacts_generated_in_run": [],
        "run_dir": str(run_dir),
    }


def init_run_manifest(
    *,
    cfg: Dict[str, Any],
    run_id: str,
    config_path: str,
    config_snapshot_path: str,
    run_dir: Path,
) -> Dict[str, Any]:
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    dataset_path = ""
    processed_file = str(paths_cfg.get("processed_stream_file", "")).strip()
    processed_dir = str(paths_cfg.get("processed_stream_dir", "")).strip()
    if processed_file:
        dataset_path = processed_file
    elif processed_dir:
        dataset_path = processed_dir
    dataset_sha = ""
    p = Path(dataset_path) if dataset_path else None
    if p and p.is_file():
        dataset_sha = sha256_file(p)
    repo_dir = Path(__file__).resolve().parents[1]
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    status_porcelain = git_cmd(repo_dir, ["status", "--short"])
    return {
        "run_id": str(run_id),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "git_branch_name": git_cmd(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_commit_sha": git_cmd(repo_dir, ["rev-parse", "HEAD"]),
        "git_status_porcelain": status_porcelain.splitlines(),
        "git_is_dirty": bool(status_porcelain.strip()),
        "mode": str(cfg.get("mode", "")),
        "experiment_name": str(cfg.get("experiment_name", "")),
        "benchmark_alias": str(data_cfg.get("stream_name", "")),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": str(dataset_sha),
        "tokenizer_path_or_name": str(model_cfg.get("hf_model_name_or_path", "")),
        "model_path_or_name": str(model_cfg.get("hf_model_name_or_path", "")),
        "lora_config": cfg.get("lora", {}),
        "train_config": cfg.get("train", {}),
        "modules_config": cfg.get("modules", {}),
        "run_dir": str(run_dir),
        "artifacts_generated_in_run": [],
    }


def finalize_run_manifest(manifest: Dict[str, Any], *, status: str, error: str = "") -> None:
    manifest["status"] = str(status)
    manifest["end_time"] = datetime.now(timezone.utc).isoformat()
    if error:
        manifest["error"] = str(error)


def manifest_add_artifact(manifest: Dict[str, Any], path: Path) -> None:
    artifacts = manifest.get("artifacts_generated_in_run")
    if not isinstance(artifacts, list):
        artifacts = []
        manifest["artifacts_generated_in_run"] = artifacts
    sp = str(path)
    if sp not in artifacts:
        artifacts.append(sp)


def collect_overfit_stale_artifacts(out_dir: Path) -> List[Path]:
    stale: List[Path] = []
    fixed = [
        out_dir / "lora_ckpt",
        out_dir / "overfit_state.json",
        out_dir / "overfit8_steps.csv",
        out_dir / "first_token_margin_over_time.csv",
        out_dir / "first_token_margin_aggregate_by_step.csv",
        out_dir / "prefix_rollout_summary.csv",
        out_dir / "decode_ablation.csv",
        out_dir / "decode_ablation.json",
        out_dir / "first_token_gold_source_compare.csv",
    ]
    stale.extend([p for p in fixed if p.exists()])
    for p in sorted(out_dir.glob("step_*_failure_*.json")):
        stale.append(p)
    for p in sorted(out_dir.glob("step_*_failure_*.csv")):
        stale.append(p)
    for p in sorted(out_dir.glob("step_*_first_token_audit.json")):
        stale.append(p)
    for p in sorted(out_dir.glob("step_*_first_token_gold_source_compare.json")):
        stale.append(p)
    for p in sorted(out_dir.glob("step_*_prefix_rollout.json")):
        stale.append(p)
    for p in sorted(out_dir.glob("step_*_predictions.json")):
        stale.append(p)
    seen = set()
    out: List[Path] = []
    for p in stale:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out
