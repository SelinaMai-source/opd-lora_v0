"""v3c matched-budget matrix. Phase 1 seed 456, then R/P seeds, then U seeds."""
from __future__ import annotations

from typing import Dict, List, Tuple

LETTERS = {
    "R": "random",
    "U": "compat_uniform",
    "P": "true_prototype",
}
OBJECTIVES = {"ce": 0.0, "opsd": 0.2}
STREAMS = {
    "instrdialog": (
        "benchmark/instrdialog",
        "citb_cl_dialogue_tasks_train50_eval10_multiref.json",
    ),
    "instrdialogpp": (
        "benchmark/instrdialogpp",
        "citb_cl_38_random_tasks_train50_eval10_multiref.json",
    ),
}
CONTENT = {
    "R": "全历史记忆均匀回放,mixed-batch ρ=0.20, seed {seed}",
    "U": "格式兼容均匀 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20, seed {seed}",
    "P": "显式 L2 余弦原型 Related + train-holdout 退化 Degraded,mixed-batch ρ=0.20, seed {seed}",
}


def run_name(letter: str, objective: str, stream: str, seed: int) -> str:
    return f"v3c_{letter}_{objective}_{stream}_seed{int(seed)}"


def phase1() -> List[str]:
    names: List[str] = []
    for stream in ("instrdialog", "instrdialogpp"):
        for objective in ("ce", "opsd"):
            for letter in ("R", "U", "P"):
                names.append(run_name(letter, objective, stream, 456))
    return names


def phase2() -> List[str]:
    """R and P at seeds 789 and 42. Seed 456 of those cells is phase 1."""
    names: List[str] = []
    for stream in ("instrdialog", "instrdialogpp"):
        for objective in ("ce", "opsd"):
            for letter in ("R", "P"):
                for seed in (789, 42):
                    names.append(run_name(letter, objective, stream, seed))
    return names


def phase3() -> List[str]:
    names: List[str] = []
    for stream in ("instrdialog", "instrdialogpp"):
        for objective in ("ce", "opsd"):
            for seed in (789, 42):
                names.append(run_name("U", objective, stream, seed))
    return names


def all_runs() -> List[str]:
    return phase1() + phase2() + phase3()


def parse_name(name: str) -> Dict[str, object]:
    # v3c_{R|U|P}_{ce|opsd}_{instrdialog|instrdialogpp}_seed{N}
    if not name.startswith("v3c_"):
        raise ValueError(name)
    body = name[len("v3c_") :]
    letter, objective, rest = body.split("_", 2)
    stream, seed_text = rest.rsplit("_seed", 1)
    if letter not in LETTERS or objective not in OBJECTIVES or stream not in STREAMS:
        raise ValueError(name)
    return {
        "letter": letter,
        "strategy": LETTERS[letter],
        "objective": objective,
        "lambda_opsd": OBJECTIVES[objective],
        "stream": stream,
        "seed": int(seed_text),
    }


def build_config(name: str) -> dict:
    info = parse_name(name)
    stream_dir, stream_file = STREAMS[str(info["stream"])]
    return {
        "experiment_name": name.rsplit("_seed", 1)[0],
        "mode": "baseline",
        "seed": int(info["seed"]),
        "baseline_name": "ce_opsd_replay",
        "paths": {
            "project_root": ".",
            "processed_stream_dir": stream_dir,
            "processed_stream_file": stream_file,
            "assets_dir": "assets",
            "results_dir": "/root/autodl-tmp/rp_lora_v3_runs",
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
        "ce_opsd": {"lambda_opsd": float(info["lambda_opsd"])},
        "output": {
            "run_name": name,
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
            "strategy": info["strategy"],
            "replay_ratio": 0.2,
            "buffer_k": 10,
            "neighbor_ratio": 0.1,
            "forgotten_ratio": 0.1,
            "transfer_slot": "neighbor",
            "coverage_dedup": False,
            "monitor": "train_holdout",
            "holdout_n": 5,
            "holdout_seed": 456,
            "budget_policy": "unified",
            "ratio_definition": "mixed_batch",
            "target_fraction": 0.20,
        },
        "teacher_proto": {"mode": "none"},
    }


def phase_of(name: str) -> int:
    if name in phase1():
        return 1
    if name in phase2():
        return 2
    if name in phase3():
        return 3
    raise KeyError(name)


def expected_counts() -> Tuple[int, int, int, int]:
    return len(phase1()), len(phase2()), len(phase3()), len(all_runs())
