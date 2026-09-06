#!/usr/bin/env python3
"""Audit train multi-reference duplicates and write NEW *_multiref.json streams.

Loads existing CITB-used benchmark streams via core.data.load_continual_stream,
merges train by (instruction, input) with unique-keep-order outputs, leaves eval
unchanged, and writes sibling files. Does not overwrite CITB-used json. CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.data import (
    Example,
    _unique_keep_order,
    example_golds,
    load_continual_stream,
)

GroupKey = Tuple[str, str]

STREAMS = (
    {
        "name": "InstrDialog",
        "src_dir": REPO_ROOT / "benchmark" / "instrdialog",
        "src_file": "citb_cl_dialogue_tasks_train50_eval10.json",
        "dst_file": "citb_cl_dialogue_tasks_train50_eval10_multiref.json",
        "new_version": "citb_instrdialog_train50_eval10_multiref_v3",
    },
    {
        "name": "InstrDialog++",
        "src_dir": REPO_ROOT / "benchmark" / "instrdialogpp",
        "src_file": "citb_cl_38_random_tasks_train50_eval10.json",
        "dst_file": "citb_cl_38_random_tasks_train50_eval10_multiref.json",
        "new_version": "citb_instrdialogpp_train50_eval10_multiref_v3",
    },
)


def _load_stream(src_dir: Path, src_file: str):
    return load_continual_stream(
        mode="baseline",
        sample_stream_path=None,
        processed_stream_dir=str(src_dir),
        processed_stream_file=src_file,
    )


def _gold_count_dist(counts: Iterable[int]) -> Dict[int, int]:
    return dict(sorted(Counter(counts).items()))


def _fmt_dist(dist: Dict[int, int]) -> str:
    return ", ".join(f"{k}gold={v}" for k, v in dist.items())


def audit_train(examples: Sequence[Example]) -> Dict[str, Any]:
    groups: Dict[GroupKey, List[str]] = defaultdict(list)
    for ex in examples:
        key = (ex.instruction, ex.input)
        groups[key] = _unique_keep_order(groups[key] + example_golds(ex))
    gold_counts = [len(golds) for golds in groups.values()]
    n_multi = sum(1 for c in gold_counts if c > 1)
    return {
        "rows": len(examples),
        "groups": len(groups),
        "multi_groups": n_multi,
        "gold_count_dist": _gold_count_dist(gold_counts),
    }


def audit_eval(examples: Sequence[Example]) -> Dict[str, Any]:
    gold_counts = [len(example_golds(ex)) for ex in examples]
    n_multi = sum(1 for c in gold_counts if c > 1)
    return {
        "rows": len(examples),
        "multi_gold_rows": n_multi,
        "gold_count_dist": _gold_count_dist(gold_counts),
    }


def merge_train(examples: Sequence[Example]) -> List[Dict[str, Any]]:
    groups: Dict[GroupKey, List[str]] = {}
    order: List[GroupKey] = []
    for ex in examples:
        key = (ex.instruction, ex.input)
        golds = example_golds(ex)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key] = _unique_keep_order(groups[key] + golds)
    rows: List[Dict[str, Any]] = []
    for key in order:
        golds = groups[key] or [""]
        rows.append(
            {
                "instruction": key[0],
                "input": key[1],
                "output": golds[0],
                "outputs": golds,
            }
        )
    return rows


def write_multiref(
    src_path: Path,
    dst_path: Path,
    merged_trains: Sequence[Sequence[Dict[str, Any]]],
    new_version: str,
) -> None:
    if dst_path.resolve() == src_path.resolve():
        raise ValueError(f"Refusing to overwrite source stream: {src_path}")
    with src_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if "stream" not in raw or not isinstance(raw["stream"], list):
        raise ValueError(f"Invalid stream json: {src_path}")
    if len(raw["stream"]) != len(merged_trains):
        raise ValueError(
            f"Segment count mismatch for {src_path}: "
            f"raw={len(raw['stream'])} merged={len(merged_trains)}"
        )
    raw["version"] = new_version
    for seg, train_rows in zip(raw["stream"], merged_trains):
        seg["train"] = list(train_rows)
        # eval left as loaded from the source file
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")


def process_stream(spec: Dict[str, Any]) -> Dict[str, Any]:
    src_dir = spec["src_dir"]
    src_path = src_dir / spec["src_file"]
    dst_path = src_dir / spec["dst_file"]
    if not src_path.exists():
        raise FileNotFoundError(f"Source stream not found: {src_path}")
    if dst_path.resolve() == src_path.resolve():
        raise ValueError(f"Destination must differ from source: {src_path}")

    stream = _load_stream(src_dir, spec["src_file"])
    train_before: List[Example] = []
    eval_all: List[Example] = []
    merged_trains: List[List[Dict[str, Any]]] = []
    for seg in stream.stream:
        train_before.extend(seg.train)
        eval_all.extend(seg.eval)
        merged_trains.append(merge_train(seg.train))

    train_audit = audit_train(train_before)
    eval_audit = audit_eval(eval_all)
    after_rows = sum(len(rows) for rows in merged_trains)
    after_multi = sum(1 for rows in merged_trains for r in rows if len(r.get("outputs") or []) > 1)

    print(f"=== {spec['name']} ===")
    print(f"src: {src_path}")
    print(f"src_version: {stream.version}")
    print(f"segments: {len(stream.stream)}")
    print(
        f"train: rows={train_audit['rows']} groups={train_audit['groups']} "
        f"multi_target_groups={train_audit['multi_groups']} "
        f"gold_count_dist={{{_fmt_dist(train_audit['gold_count_dist'])}}}"
    )
    print(
        f"eval: rows={eval_audit['rows']} multi_gold_rows={eval_audit['multi_gold_rows']} "
        f"gold_count_dist={{{_fmt_dist(eval_audit['gold_count_dist'])}}}"
    )
    print(f"train before/after rows: {train_audit['rows']} -> {after_rows}")
    print(f"dst_version: {spec['new_version']}")

    write_multiref(src_path, dst_path, merged_trains, spec["new_version"])
    print(f"wrote: {dst_path}")

    reloaded = _load_stream(src_dir, spec["dst_file"])
    reload_train = [ex for seg in reloaded.stream for ex in seg.train]
    reload_eval = [ex for seg in reloaded.stream for ex in seg.eval]
    reload_train_audit = audit_train(reload_train)
    reload_eval_audit = audit_eval(reload_eval)
    if reloaded.version != spec["new_version"]:
        raise RuntimeError(f"Version mismatch after reload: {reloaded.version}")
    if reload_train_audit["rows"] != after_rows:
        raise RuntimeError(
            f"Reloaded train rows {reload_train_audit['rows']} != written {after_rows}"
        )
    if reload_train_audit["groups"] != after_rows:
        raise RuntimeError("Reloaded train still has duplicate (instruction, input) keys")
    if reload_eval_audit["rows"] != eval_audit["rows"]:
        raise RuntimeError(
            f"Eval rows changed: {eval_audit['rows']} -> {reload_eval_audit['rows']}"
        )
    if reload_eval_audit["multi_gold_rows"] != eval_audit["multi_gold_rows"]:
        raise RuntimeError("Eval multi-gold row count changed after write")
    print(
        f"verify: train_rows={reload_train_audit['rows']} "
        f"train_multi_output_rows={after_multi} "
        f"eval_rows={reload_eval_audit['rows']} "
        f"eval_multi_gold={reload_eval_audit['multi_gold_rows']} "
        f"version={reloaded.version}"
    )
    print()
    return {
        "name": spec["name"],
        "src": str(src_path),
        "dst": str(dst_path),
        "src_version": stream.version,
        "dst_version": spec["new_version"],
        "segments": len(stream.stream),
        "train_rows_before": train_audit["rows"],
        "train_rows_after": after_rows,
        "train_groups": train_audit["groups"],
        "train_multi_target_groups": train_audit["multi_groups"],
        "train_gold_count_dist": train_audit["gold_count_dist"],
        "eval_rows": eval_audit["rows"],
        "eval_multi_gold_rows": eval_audit["multi_gold_rows"],
        "eval_gold_count_dist": eval_audit["gold_count_dist"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    summaries = [process_stream(spec) for spec in STREAMS]
    print("=== summary ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
