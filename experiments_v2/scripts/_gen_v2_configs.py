#!/usr/bin/env python3
"""Write remaining v2 seed-456 yaml configs (idempotent)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path("/root/autodl-tmp/rp_lora_v0")
CFG = ROOT / "configs"


def base(stream: str) -> dict:
    if stream == "instrdialog":
        ddir, dfile = "benchmark/instrdialog", "citb_cl_dialogue_tasks_train50_eval10_multiref.json"
        exp = "instrdialog"
    else:
        ddir, dfile = "benchmark/instrdialogpp", "citb_cl_38_random_tasks_train50_eval10_multiref.json"
        exp = "instrdialogpp"
    return {
        "experiment_name": "",
        "mode": "baseline",
        "seed": 456,
        "baseline_name": "",
        "paths": {
            "project_root": ".",
            "processed_stream_dir": ddir,
            "processed_stream_file": dfile,
            "assets_dir": "assets",
            "results_dir": "results",
        },
        "data": {
            "stream_format": "citb_processed",
            "split": "default",
            "max_segments": -1,
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
        "ce_opsd": {"lambda_opsd": 0.2},
        "output": {
            "run_name": "",
            "save_every_segment": True,
            "write_tables": True,
            "write_logs": True,
            "tracking": {"use_wandb": False},
            "save_final_adapter": True,
        },
        "eval": {"debug_max_examples": 0},
        "eval_normalization": {"format_normalize": True},
    }


def dump(name: str, cfg: dict) -> None:
    path = CFG / name
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(path)


def replay(stream: str, strategy: str, teacher_mode: str = "none", smoke: bool = False) -> None:
    tag = "instrdialog" if stream == "instrdialog" else "instrdialogpp"
    cfg = base(stream)
    cfg["baseline_name"] = "ce_opsd_replay"
    cfg["ce_opsd"]["lambda_opsd"] = 0.2
    cfg["replay"] = {
        "strategy": strategy,
        "replay_ratio": 0.2,
        "buffer_k": 10,
        "neighbor_ratio": 0.1,
        "forgotten_ratio": 0.1,
    }
    cfg["teacher_proto"] = {"mode": teacher_mode}
    if strategy == "uniform":
        family = "v2_rand_replay_opsd"
        exp = f"v2_rand_replay_opsd_{tag}"
        run = f"v2_rand_replay_opsd_{tag}_seed456"
    elif teacher_mode != "none":
        family = "v2_proto_teacher_opsd"
        exp = f"v2_proto_teacher_opsd_{teacher_mode}_{tag}"
        run = f"v2_proto_teacher_opsd_{teacher_mode}_{tag}_seed456"
    else:
        family = "v2_proto_replay_opsd"
        exp = f"v2_proto_replay_opsd_{tag}"
        run = f"v2_proto_replay_opsd_{tag}_seed456"
    if smoke:
        cfg["data"]["max_segments"] = 2
        run = f"{run}_smoke2seg"
        exp = f"{exp}_smoke2seg"
    cfg["experiment_name"] = exp
    cfg["output"]["run_name"] = run
    dump(f"{run}.yaml", cfg)


def metric(stream: str, mu: float) -> None:
    tag = "instrdialog" if stream == "instrdialog" else "instrdialogpp"
    mu_tag = {0.05: "m005", 0.1: "m01", 0.2: "m02"}[mu]
    cfg = base(stream)
    cfg["baseline_name"] = "metric_opsd"
    cfg["ce_opsd"]["lambda_opsd"] = 0.2
    cfg["metric_opsd"] = {
        "mu": mu,
        "every_n_batches": 6,
        "n_candidates": 4,
        "pref_beta": 1.0,
    }
    run = f"v2_metric_opsd_{tag}_{mu_tag}_seed456"
    cfg["experiment_name"] = f"v2_metric_opsd_{tag}"
    cfg["output"]["run_name"] = run
    dump(f"{run}.yaml", cfg)


def main() -> None:
    for stream in ("instrdialog", "instrdialogpp"):
        replay(stream, "uniform")
        replay(stream, "proto", "none")
        replay(stream, "proto", "random")
        replay(stream, "proto", "matched")
        for mu in (0.05, 0.1, 0.2):
            metric(stream, mu)
    replay("instrdialog", "proto", "none", smoke=True)


if __name__ == "__main__":
    main()
