#!/usr/bin/env python3
"""Write the old-protocol 12 and the v3h 2x2 configs. Does not launch training."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path("/root/research-repos/opd-lora_v3")
OUT = ROOT / "configs"
RESULTS = "/root/autodl-tmp/rp_lora_v3_runs"

STREAMS = {
    "instrdialog": {
        "dir": "benchmark/instrdialog",
        "file": "citb_cl_dialogue_tasks_train50_eval10_multiref.json",
    },
    "instrdialogpp": {
        "dir": "benchmark/instrdialogpp",
        "file": "citb_cl_38_random_tasks_train50_eval10_multiref.json",
    },
}

METHODS = {
    "rand_ce": {"strategy": "uniform", "lambda_opsd": 0.0},
    "proto_ce": {"strategy": "proto", "lambda_opsd": 0.0},
    "rand_opsd": {"strategy": "uniform", "lambda_opsd": 0.2},
    "proto_opsd": {"strategy": "proto", "lambda_opsd": 0.2},
}


def build(
    *,
    prefix: str,
    method: str,
    stream: str,
    seed: int,
    monitor: str,
    max_segments: int,
) -> dict:
    spec = METHODS[method]
    data = STREAMS[stream]
    run_name = f"{prefix}_{method}_{stream}_seed{seed}"
    if max_segments > 0:
        run_name += f"_smoke{max_segments}seg"
    cfg = {
        "experiment_name": f"{prefix}_{method}_{stream}",
        "mode": "baseline",
        "seed": int(seed),
        "baseline_name": "ce_opsd_replay",
        "paths": {
            "project_root": ".",
            "processed_stream_dir": data["dir"],
            "processed_stream_file": data["file"],
            "assets_dir": "assets",
            "results_dir": RESULTS,
        },
        "data": {
            "stream_format": "citb_processed",
            "split": "default",
            "max_segments": int(max_segments),
            "max_train_examples_per_segment": -1,
            "max_eval_examples_per_segment": -1,
        },
        "model": {
            "backbone_name": "llama31-8b-instruct",
            "hf_model_name_or_path": "assets/pretrained/Meta-Llama-3.1-8B-Instruct",
            "torch_dtype": "bfloat16",
            "device": "auto",
            "mask_eos_token_in_labels": False,
            "mask_all_special_tokens_in_labels": False,
        },
        "lora": {
            "enabled": True,
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
        },
        "train": {
            "epochs_per_segment": 1,
            "batch_size": 2,
            "lr": 0.0002,
            "weight_decay": 0.0,
            "grad_clip_norm": 1.0,
            "log_every": 50,
        },
        "opsd": {
            "divergence": "jsd",
            "beta": 0.5,
            "rollout_temperature": 1.0,
            "rollout_top_p": 1.0,
            "rollout_max_new_tokens": 0,
            "teacher_format_constraint": True,
        },
        "ce_opsd": {"lambda_opsd": float(spec["lambda_opsd"])},
        "output": {
            "run_name": run_name,
            "save_every_segment": True,
            "save_segment_adapters": False,
            "write_tables": True,
            "write_logs": True,
            "tracking": {"use_wandb": False},
            "save_final_adapter": True,
        },
        "eval": {"debug_max_examples": 0},
        "eval_normalization": {"format_normalize": True},
        "replay": {
            "strategy": spec["strategy"],
            "replay_ratio": 0.2,
            "buffer_k": 10,
            "neighbor_ratio": 0.1,
            "forgotten_ratio": 0.1,
            "transfer_slot": "neighbor",
            "coverage_dedup": False,
            "monitor": monitor,
        },
        "teacher_proto": {"mode": "none"},
    }
    if monitor == "train_holdout":
        cfg["replay"]["holdout_n"] = 5
        cfg["replay"]["holdout_seed"] = int(seed)
    return cfg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for method in ("rand_ce", "proto_ce", "rand_opsd"):
        for stream in ("instrdialog", "instrdialogpp"):
            for seed in (789, 42):
                cfg = build(
                    prefix="v3",
                    method=method,
                    stream=stream,
                    seed=seed,
                    monitor="eval_split_monitor",
                    max_segments=-1,
                )
                path = OUT / f"{cfg['output']['run_name']}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
                written.append(path.name)
    for method in ("rand_ce", "proto_ce", "rand_opsd", "proto_opsd"):
        for stream in ("instrdialog", "instrdialogpp"):
            cfg = build(
                prefix="v3h",
                method=method,
                stream=stream,
                seed=456,
                monitor="train_holdout",
                max_segments=-1,
            )
            path = OUT / f"{cfg['output']['run_name']}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            written.append(path.name)
    smoke = build(
        prefix="v3h",
        method="proto_opsd",
        stream="instrdialog",
        seed=456,
        monitor="train_holdout",
        max_segments=2,
    )
    smoke_path = OUT / f"{smoke['output']['run_name']}.yaml"
    smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False, allow_unicode=True), encoding="utf-8")
    written.append(smoke_path.name)
    print(f"wrote {len(written)} configs")
    for name in written:
        print(name)


if __name__ == "__main__":
    main()
