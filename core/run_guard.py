"""Refuse to overwrite an existing run directory.

A completed run whose config fingerprint matches the yaml is a skip (exit 0).
Anything else already present is left untouched (exit 3 mismatch, exit 4
incomplete).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def run_directory(results_dir: str, run_name: str) -> Optional[Path]:
    name = str(run_name or "").strip()
    if not name:
        return None
    return Path(results_dir) / "runs" / name


def fingerprint(cfg: Dict[str, Any]) -> Dict[str, Any]:
    replay = cfg.get("replay") if isinstance(cfg.get("replay"), dict) else {}
    output = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    data = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    ce = cfg.get("ce_opsd") if isinstance(cfg.get("ce_opsd"), dict) else {}
    opsd = cfg.get("opsd") if isinstance(cfg.get("opsd"), dict) else {}
    model = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    monitor = str(replay.get("monitor", "eval_split_monitor")).strip().lower()
    order = data.get("segment_order") or []
    out = {
        "seed": int(cfg.get("seed", 0)),
        "baseline_name": str(cfg.get("baseline_name", "")),
        "run_name": str(output.get("run_name", "")),
        "strategy": str(replay.get("strategy", "")),
        "monitor": monitor,
        "lambda_opsd": float(ce.get("lambda_opsd", 0.0)),
        "stream_file": str(paths.get("processed_stream_file", "")),
        "stream_dir": str(paths.get("processed_stream_dir", "")),
        "max_segments": int(data.get("max_segments", -1)),
        "replay_ratio": float(replay.get("replay_ratio", 0.0)),
        "neighbor_ratio": float(replay.get("neighbor_ratio", 0.0)),
        "forgotten_ratio": float(replay.get("forgotten_ratio", 0.0)),
        "buffer_k": int(replay.get("buffer_k", replay.get("k", 10))),
        "budget_policy": str(replay.get("budget_policy", "legacy")).strip().lower(),
        "ratio_definition": str(replay.get("ratio_definition", "")).strip().lower(),
        "target_fraction": float(
            replay.get("target_fraction", replay.get("replay_ratio", 0.0))
        ),
        "student_tokens": str(opsd.get("student_tokens", "on_policy")).strip().lower(),
        "segment_order": [int(x) for x in order],
        "init_adapter_path": str(model.get("init_adapter_path", "")).strip(),
        "format_normalize": bool(
            (cfg.get("eval_normalization") or {}).get("format_normalize", False)
            if isinstance(cfg.get("eval_normalization"), dict)
            else False
        ),
    }
    if monitor == "train_holdout":
        out["holdout_n"] = int(replay.get("holdout_n", 5))
        out["holdout_seed"] = int(replay.get("holdout_seed", cfg.get("seed", 0)))
    return out


def fingerprints_match(saved: Dict[str, Any], current: Dict[str, Any]) -> bool:
    left = fingerprint(saved)
    right = fingerprint(current)
    return left == right


def load_snapshot(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}
    return data


def refuse_or_skip_existing_run(results_dir: str, run_name: str, cfg: Dict[str, Any]) -> None:
    directory = run_directory(results_dir, run_name)
    if directory is None or not directory.exists():
        return
    metrics = directory / "final_metrics.json"
    if metrics.is_file():
        saved = load_snapshot(directory / "config_snapshot.yaml")
        if saved and fingerprints_match(saved, cfg):
            print(f"SKIP existing matching run {directory}", flush=True)
            raise SystemExit(0)
        print(f"REFUSE overwrite mismatched run {directory}", flush=True)
        raise SystemExit(3)
    print(f"REFUSE overwrite incomplete run directory {directory}", flush=True)
    raise SystemExit(4)


def main_guard_from_cfg(cfg: Dict[str, Any]) -> None:
    paths = cfg.get("paths") if isinstance(cfg.get("paths"), dict) else {}
    output = cfg.get("output") if isinstance(cfg.get("output"), dict) else {}
    results_dir = str(paths.get("results_dir", "results"))
    run_name = str(output.get("run_name", ""))
    refuse_or_skip_existing_run(results_dir, run_name, cfg)


def explain_exit(code: int) -> str:
    return {
        0: "skipped_matching_final_metrics",
        3: "refused_config_mismatch",
        4: "refused_incomplete_directory",
    }.get(int(code), f"exit_{code}")
