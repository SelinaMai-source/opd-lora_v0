#!/usr/bin/env python3
"""Export full greedy eval prediction/gold CSVs for v0 SFT adapters.

Loads Instruct base + the given LoRA adapter, runs evaluate_stream on all
segments (same greedy path as training tables), and writes every eval case.

eval.debug_max_examples: 0 means no dump truncation. This script always
requests the full set; it will not silently fall back to a 50-example subset.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CSV_COLUMNS = [
    "dataset",
    "segment_id",
    "segment_name",
    "example_idx",
    "instruction",
    "input",
    "gold",
    "prediction",
    "strict_match",
    "task_aware_match",
    "task_score_type",
    "token_f1",
    "rouge_l",
    "bleu",
    "lcs_overlap",
]

JOBS = {
    "instrdialog": {
        "dataset": "instrdialog",
        "config": "configs/v0_sequential_instrdialog.yaml",
        "adapter": "results/runs/v0_sft_instrdialog/final_adapter",
        "csv_name": "instrdialog_pred_gold.csv",
    },
    "instrdialogpp": {
        "dataset": "instrdialog++",
        "config": "configs/v0_sequential_instrdialogpp.yaml",
        "adapter": "results/runs/v0_sft_instrdialogpp/final_adapter",
        "csv_name": "instrdialogpp_pred_gold.csv",
    },
}


def _gpu_compute_apps() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
            text=True,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"nvidia-smi failed; cannot confirm GPU is free: {exc}") from exc


def _foreign_gpu_rows() -> List[str]:
    raw = _gpu_compute_apps()
    self_pid = os.getpid()
    rows: List[str] = []
    for ln in raw.splitlines()[1:]:
        line = ln.strip()
        if not line:
            continue
        pid_str = line.split(",", 1)[0].strip()
        try:
            pid = int(pid_str)
        except ValueError:
            rows.append(line)
            continue
        if pid == self_pid:
            continue
        rows.append(line)
    return rows


def _assert_gpu_free() -> None:
    rows = _foreign_gpu_rows()
    if rows:
        print("GPU is occupied; refusing to start full re-eval (will not use a 50-example subset).", file=sys.stderr)
        print(_gpu_compute_apps(), file=sys.stderr)
        raise SystemExit(2)


def _example_to_row(dataset: str, ex: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset": dataset,
        "segment_id": ex.get("source_segment_id"),
        "segment_name": ex.get("source_segment_name"),
        "example_idx": ex.get("source_example_idx"),
        "instruction": ex.get("instruction", ""),
        "input": ex.get("input_text", ""),
        "gold": ex.get("gold_output", ""),
        "prediction": ex.get("raw_generated_output", ""),
        "strict_match": bool(ex.get("strict_match")),
        "task_aware_match": bool(ex.get("task_aware_match")),
        "task_score_type": ex.get("task_score_type", ""),
        "token_f1": ex.get("token_f1"),
        "rouge_l": ex.get("rouge_l"),
        "bleu": ex.get("bleu"),
        "lcs_overlap": ex.get("lcs_overlap"),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _count_expected_eval(stream: Any) -> int:
    return int(sum(len(seg.eval) for seg in stream.stream))


def _load_stream_from_cfg(cfg: Dict[str, Any]) -> Any:
    from core.data import load_continual_stream

    paths = cfg.get("paths", {}) if isinstance(cfg.get("paths", {}), dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    return load_continual_stream(
        mode=str(cfg.get("mode", "baseline")),
        sample_stream_path=None,
        processed_stream_dir=str(paths.get("processed_stream_dir", "data/processed")),
        processed_stream_file=str(paths.get("processed_stream_file", "")),
        processed_stream_name=str(data_cfg.get("stream_name", "")).strip(),
        auto_prepare_processed=bool(data_cfg.get("auto_prepare_processed", False)),
        seed=int(cfg.get("seed", 0)),
        max_segments=int(data_cfg.get("max_segments", -1)),
        max_train_examples_per_segment=int(data_cfg.get("max_train_examples_per_segment", -1)),
        max_eval_examples_per_segment=int(data_cfg.get("max_eval_examples_per_segment", -1)),
    )


def export_one(
    *,
    job_key: str,
    backbone: Any,
    lora: Any,
    out_dir: Path,
) -> Path:
    from core.evaluate import evaluate_stream
    from core.utils import load_yaml_config

    spec = JOBS[job_key]
    cfg_path = REPO_ROOT / spec["config"]
    adapter_path = REPO_ROOT / spec["adapter"]
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    cfg = load_yaml_config(str(cfg_path))
    cfg["__config_path__"] = str(cfg_path.resolve())
    stream = _load_stream_from_cfg(cfg)
    expected_n = _count_expected_eval(stream)
    print(
        f"[{spec['dataset']}] segments={len(stream.stream)} expected_eval={expected_n} "
        f"adapter={adapter_path}",
        flush=True,
    )

    lora.load_adapter_checkpoint(str(adapter_path))
    print(f"[{spec['dataset']}] loaded adapter; running evaluate_stream (full stream)...", flush=True)

    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    eval_max_new_tokens = int(model_cfg.get("gen_max_new_tokens", 64))
    normalization_cfg = cfg.get("eval_normalization", {}) if isinstance(cfg.get("eval_normalization", {}), dict) else {}

    last_seg_id = int(stream.stream[-1].segment_id) if stream.stream else 0
    metrics = evaluate_stream(
        model=backbone,
        segments_seen=list(stream.stream),
        max_new_tokens=eval_max_new_tokens,
        router=None,
        lora_bank=None,
        segment_id=last_seg_id,
        normalization_cfg=normalization_cfg,
        save_debug_examples_dir=None,
        debug_max_examples=0,
        return_eval_examples=True,
    )
    extra = metrics.get("extra", {}) if isinstance(metrics.get("extra"), dict) else {}
    examples = extra.get("eval_examples") or []
    if not examples:
        raise RuntimeError(
            f"[{spec['dataset']}] evaluate_stream returned 0 examples; refusing to write an empty/subset CSV."
        )
    if len(examples) != expected_n:
        raise RuntimeError(
            f"[{spec['dataset']}] expected {expected_n} eval cases, got {len(examples)}; "
            "refusing to write a truncated CSV."
        )

    rows = [_example_to_row(spec["dataset"], ex) for ex in examples]
    out_path = out_dir / spec["csv_name"]
    _write_csv(out_path, rows)
    print(
        f"[{spec['dataset']}] wrote {len(rows)} rows -> {out_path} "
        f"seen_avg_score={metrics.get('seen_avg_score')} "
        f"rouge_l_mean={metrics.get('rouge_l_mean')} "
        f"bleu_mean={metrics.get('bleu_mean')}",
        flush=True,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full v0 SFT pred/gold CSVs")
    parser.add_argument(
        "--dataset",
        choices=["both", "instrdialog", "instrdialogpp"],
        default="both",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "experiments_v1" / "v0_LoRA_SFT" / "predictions"),
    )
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/autodl-tmp/hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/hf_cache/hub")
    os.environ.setdefault("TORCH_HOME", "/root/autodl-tmp/torch_cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    _assert_gpu_free()

    from core.models.base_model import build_backbone
    from core.models.lora_wrapper import build_lora_wrapper
    from core.utils import load_yaml_config, set_seed

    keys = ["instrdialog", "instrdialogpp"] if args.dataset == "both" else [args.dataset]
    first_cfg = load_yaml_config(str(REPO_ROOT / JOBS[keys[0]]["config"]))
    set_seed(int(first_cfg.get("seed", 123)))

    print("Building Instruct backbone...", flush=True)
    backbone = build_backbone(first_cfg.get("model", {}), mode="baseline", seed=int(first_cfg.get("seed", 123)))
    print("Building LoRA wrapper...", flush=True)
    lora = build_lora_wrapper(backbone, first_cfg.get("lora", {}))

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    for i, key in enumerate(keys):
        _assert_gpu_free()
        written.append(export_one(job_key=key, backbone=backbone, lora=lora, out_dir=out_dir))
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"Finished {key} ({i + 1}/{len(keys)}).", flush=True)

    print("Done.", flush=True)
    for p in written:
        n_lines = sum(1 for _ in p.open(encoding="utf-8"))
        print(f"  {p}  file_lines={n_lines}  data_rows={n_lines - 1}")


if __name__ == "__main__":
    main()
