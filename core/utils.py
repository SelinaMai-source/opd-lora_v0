import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a python dict."""
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"Config must be a mapping/dict, got {type(obj)} from {path}")
    return obj


def ensure_dir(path: str) -> str:
    """Create directory if not exists; return normalized path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(Path(path))


def timestamp(compact: bool = True) -> str:
    """Return a human-friendly timestamp."""
    t = time.localtime()
    if compact:
        return time.strftime("%Y%m%d_%H%M%S", t)
    return time.strftime("%Y-%m-%d %H:%M:%S", t)


def join_path(*parts: str) -> str:
    return str(Path(*parts))


def set_seed(seed: int) -> None:
    """Set python/numpy seeds (torch handled in model code if available)."""
    random.seed(seed)
    np.random.seed(seed)


def save_json(path: str, obj: Any, indent: int = 2) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def append_jsonl(path: str, obj: Any) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_jsonl(path: str, rows: List[Any]) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_text(path: str, text: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(str(Path(path).parent))
    if not rows:
        fieldnames = fieldnames or []
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


@dataclass
class RunPaths:
    run_id: str
    run_dir: str
    log_file: str
    metrics_json: str
    segment_metrics_csv: str


def make_run_paths(results_dir: str, experiment_name: str, run_name: str = "") -> RunPaths:
    run_id = run_name.strip() or f"{timestamp()}_{experiment_name}"
    run_dir = ensure_dir(join_path(results_dir, "runs", run_id))
    logs_dir = ensure_dir(join_path(results_dir, "logs"))
    tables_dir = ensure_dir(join_path(results_dir, "tables"))

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        log_file=join_path(logs_dir, f"{run_id}.log"),
        metrics_json=join_path(run_dir, "final_metrics.json"),
        segment_metrics_csv=join_path(tables_dir, f"{run_id}_segment_metrics.csv"),
    )


class SimpleLogger:
    """Very small logger writing both to stdout and a file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        ensure_dir(str(Path(log_path).parent))

    def log(self, msg: str) -> None:
        line = f"[{timestamp(compact=False)}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

