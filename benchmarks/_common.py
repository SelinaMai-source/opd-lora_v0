"""Shared loader for the two CITB streams. Paths come from an argument, config, or env."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from core.data import ContinualStream, example_golds, load_continual_stream

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_stream_file(
    dirname: str,
    filename: str,
    *,
    root: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Find the multiref json.

    ``OPD_BENCHMARK_ROOT`` is the directory that contains ``instrdialog`` and
    ``instrdialogpp``. A config may instead pass ``paths.processed_stream_dir``
    and ``paths.processed_stream_file``. No machine-specific disk is required.
    """
    if config:
        paths = config.get("paths") or {}
        configured_dir = paths.get("processed_stream_dir")
        configured_file = paths.get("processed_stream_file") or filename
        if configured_dir:
            candidate = Path(str(configured_dir))
            if not candidate.is_absolute():
                candidate = REPO_ROOT / candidate
            stream_file = Path(str(configured_file))
            if not stream_file.is_absolute():
                stream_file = candidate / stream_file.name
            return stream_file
    if root:
        base = Path(root)
        if base.is_file():
            return base
        return base / filename
    env = os.environ.get("OPD_BENCHMARK_ROOT", "").strip()
    if env:
        return Path(env) / dirname / filename
    return REPO_ROOT / "benchmark" / dirname / filename


def load_stream(path: Path) -> ContinualStream:
    if not path.is_file():
        raise FileNotFoundError(
            f"Benchmark file not found: {path}. Set OPD_BENCHMARK_ROOT to the directory "
            "that contains instrdialog/ and instrdialogpp/, or pass paths in the config. "
            "The json files are not stored in git."
        )
    return load_continual_stream(
        mode="baseline",
        sample_stream_path=None,
        processed_stream_dir=str(path.parent),
        processed_stream_file=path.name,
    )


def validate_dataset(stream: ContinualStream) -> Dict[str, int]:
    if not stream.stream:
        raise ValueError("stream has no segments")
    seen = set()
    n_train = 0
    n_eval = 0
    for seg in stream.stream:
        if seg.segment_id in seen:
            raise ValueError(f"duplicate segment_id {seg.segment_id}")
        seen.add(seg.segment_id)
        if not seg.train or not seg.eval:
            raise ValueError(f"segment {seg.segment_id} is missing train or eval")
        train_keys = {(ex.instruction, ex.input) for ex in seg.train}
        eval_keys = {(ex.instruction, ex.input) for ex in seg.eval}
        overlap = train_keys & eval_keys
        if overlap:
            raise ValueError(f"segment {seg.segment_id} has {len(overlap)} train/eval key overlaps")
        for ex in list(seg.train) + list(seg.eval):
            if not str(ex.instruction):
                raise ValueError(f"segment {seg.segment_id} has an empty instruction")
        for ex in seg.eval:
            if not example_golds(ex):
                raise ValueError(f"segment {seg.segment_id} eval row has no gold")
        n_train += len(seg.train)
        n_eval += len(seg.eval)
    return {"n_segments": len(stream.stream), "n_train": n_train, "n_eval": n_eval}


def metadata(name: str, dirname: str, filename: str, citation: str) -> Dict[str, str]:
    return {
        "name": name,
        "dirname": dirname,
        "filename": filename,
        "source": citation,
        "format": "citb_processed multiref; eval keeps an outputs list",
        "env": "OPD_BENCHMARK_ROOT",
    }
