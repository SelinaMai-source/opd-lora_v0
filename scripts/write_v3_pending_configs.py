#!/usr/bin/env python3
"""Write ACL pending-extension yaml. Does not launch training."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path("/root/research-repos/opd-lora_v3")
OUT = ROOT / "configs"
RESULTS = "/root/autodl-tmp/rp_lora_v3_runs"

STREAMS = {
    "instrdialog": {
        "dir": "benchmark/instrdialog",
        "file": "citb_cl_dialogue_tasks_train50_eval10_multiref.json",
        "n_seg": 19,
    },
    "instrdialogpp": {
        "dir": "benchmark/instrdialogpp",
        "file": "citb_cl_38_random_tasks_train50_eval10_multiref.json",
        "n_seg": 38,
    },
}

# python random.Random(456).shuffle(list(range(19)))
ID_SHUFFLE_456 = [4, 11, 2, 3, 16, 17, 6, 7, 5, 1, 8, 9, 0, 10, 15, 12, 13, 18, 14]
ID_REVERSE = list(range(18, -1, -1))


def proto_backbone(
    *,
    stream: str,
    run_name: str,
    experiment_name: str,
    neighbor_ratio: float,
    forgotten_ratio: float,
    replay_ratio: float,
    lambda_opsd: float,
    buffer_k: int,
    holdout: bool,
    student_tokens: str = "on_policy",
    segment_order=None,
    segment_order_name: str = "",
    baseline_name: str = "ce_opsd_replay",
    extra: dict | None = None,
) -> dict:
    data = STREAMS[stream]
    replay_monitor = "train_holdout" if holdout else "eval_split_monitor"
    cfg = {
        "experiment_name": experiment_name,
        "mode": "baseline",
        "seed": 456,
        "baseline_name": baseline_name,
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
            "student_tokens": student_tokens,
        },
        "ce_opsd": {"lambda_opsd": float(lambda_opsd)},
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
            "strategy": "proto",
            "replay_ratio": float(replay_ratio),
            "buffer_k": int(buffer_k),
            "neighbor_ratio": float(neighbor_ratio),
            "forgotten_ratio": float(forgotten_ratio),
            "transfer_slot": "neighbor",
            "coverage_dedup": False,
            "monitor": replay_monitor,
        },
        "teacher_proto": {"mode": "none"},
    }
    if holdout:
        cfg["replay"]["holdout_n"] = 5
        cfg["replay"]["holdout_seed"] = 456
    if segment_order is not None:
        cfg["data"]["segment_order"] = [int(x) for x in segment_order]
        if segment_order_name:
            cfg["data"]["segment_order_name"] = segment_order_name
    if extra:
        cfg = _deep_merge(cfg, extra)
    return cfg


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in extra.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def dump(cfg: dict) -> Path:
    path = OUT / f"{cfg['output']['run_name']}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def specs() -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for stream in ("instrdialog", "instrdialogpp"):
        for nb, fg, tag in ((0.15, 0.05, "n15f5"), (0.05, 0.15, "n5f15")):
            name = f"v3h_proto_opsd_{tag}_{stream}_seed456"
            rows.append(
                (
                    "e2",
                    proto_backbone(
                        stream=stream,
                        run_name=name,
                        experiment_name=f"v3h_proto_opsd_{tag}_{stream}",
                        neighbor_ratio=nb,
                        forgotten_ratio=fg,
                        replay_ratio=0.2,
                        lambda_opsd=0.2,
                        buffer_k=10,
                        holdout=True,
                    ),
                )
            )
        for lam, tag in ((0.05, "lam005"), (0.10, "lam010"), (0.50, "lam050")):
            name = f"v3h_proto_opsd_{tag}_{stream}_seed456"
            rows.append(
                (
                    "e4lam",
                    proto_backbone(
                        stream=stream,
                        run_name=name,
                        experiment_name=f"v3h_proto_opsd_{tag}_{stream}",
                        neighbor_ratio=0.1,
                        forgotten_ratio=0.1,
                        replay_ratio=0.2,
                        lambda_opsd=lam,
                        buffer_k=10,
                        holdout=True,
                    ),
                )
            )
        name = f"v3h_proto_opsd_offpol_{stream}_seed456"
        rows.append(
            (
                "e4off",
                proto_backbone(
                    stream=stream,
                    run_name=name,
                    experiment_name=f"v3h_proto_opsd_offpol_{stream}",
                    neighbor_ratio=0.1,
                    forgotten_ratio=0.1,
                    replay_ratio=0.2,
                    lambda_opsd=0.2,
                    buffer_k=10,
                    holdout=True,
                    student_tokens="gold",
                ),
            )
        )
        # E6 0%: disable replay, no holdout (forgotten_ratio=0).
        name = f"v3_proto_opsd_rr0_{stream}_seed456"
        rows.append(
            (
                "e6",
                proto_backbone(
                    stream=stream,
                    run_name=name,
                    experiment_name=f"v3_proto_opsd_rr0_{stream}",
                    neighbor_ratio=0.0,
                    forgotten_ratio=0.0,
                    replay_ratio=0.0,
                    lambda_opsd=0.2,
                    buffer_k=10,
                    holdout=False,
                ),
            )
        )
        for rr, tag in ((0.10, "rr10"), (0.30, "rr30"), (0.40, "rr40")):
            half = rr / 2.0
            name = f"v3h_proto_opsd_{tag}_{stream}_seed456"
            rows.append(
                (
                    "e6",
                    proto_backbone(
                        stream=stream,
                        run_name=name,
                        experiment_name=f"v3h_proto_opsd_{tag}_{stream}",
                        neighbor_ratio=half,
                        forgotten_ratio=half,
                        replay_ratio=rr,
                        lambda_opsd=0.2,
                        buffer_k=10,
                        holdout=True,
                    ),
                )
            )
        # E3 direct OPSD: existing opsd method, no CE, no replay, no holdout.
        name = f"v3_opsd_direct_{stream}_seed456"
        direct = proto_backbone(
            stream=stream,
            run_name=name,
            experiment_name=f"v3_opsd_direct_{stream}",
            neighbor_ratio=0.0,
            forgotten_ratio=0.0,
            replay_ratio=0.0,
            lambda_opsd=0.0,
            buffer_k=10,
            holdout=False,
            baseline_name="opsd",
            extra={"replay": {"strategy": "uniform", "monitor": "eval_split_monitor"}},
        )
        rows.append(("e3", direct))
        # E3 alternating CE/OPSD: existing seg_opsd, K=25, 2 epochs as designed.
        name = f"v3_seg_opsd_k25_{stream}_seed456"
        seg = proto_backbone(
            stream=stream,
            run_name=name,
            experiment_name=f"v3_seg_opsd_k25_{stream}",
            neighbor_ratio=0.0,
            forgotten_ratio=0.0,
            replay_ratio=0.0,
            lambda_opsd=0.0,
            buffer_k=10,
            holdout=False,
            baseline_name="seg_opsd",
            extra={
                "train": {"epochs_per_segment": 2},
                "seg_opsd": {"k": 25, "teacher_adapter": "teacher"},
                "replay": {"strategy": "uniform", "monitor": "eval_split_monitor", "buffer_size": 2048},
            },
        )
        rows.append(("e3", seg))
        for k, tag in ((5, "bufk5"), (20, "bufk20"), (50, "bufk50")):
            name = f"v3h_proto_opsd_{tag}_{stream}_seed456"
            rows.append(
                (
                    "e9",
                    proto_backbone(
                        stream=stream,
                        run_name=name,
                        experiment_name=f"v3h_proto_opsd_{tag}_{stream}",
                        neighbor_ratio=0.1,
                        forgotten_ratio=0.1,
                        replay_ratio=0.2,
                        lambda_opsd=0.2,
                        buffer_k=k,
                        holdout=True,
                    ),
                )
            )
        name = f"v3h_proto_opsd_allnb_{stream}_seed456"
        rows.append(
            (
                "e11",
                proto_backbone(
                    stream=stream,
                    run_name=name,
                    experiment_name=f"v3h_proto_opsd_allnb_{stream}",
                    neighbor_ratio=0.2,
                    forgotten_ratio=0.0,
                    replay_ratio=0.2,
                    lambda_opsd=0.2,
                    buffer_k=10,
                    holdout=True,
                ),
            )
        )
        name = f"v3h_proto_opsd_allfg_{stream}_seed456"
        rows.append(
            (
                "e11",
                proto_backbone(
                    stream=stream,
                    run_name=name,
                    experiment_name=f"v3h_proto_opsd_allfg_{stream}",
                    neighbor_ratio=0.0,
                    forgotten_ratio=0.2,
                    replay_ratio=0.2,
                    lambda_opsd=0.2,
                    buffer_k=10,
                    holdout=True,
                ),
            )
        )
    for tag, order, order_name in (
        ("ordrev", ID_REVERSE, "reverse_original"),
        ("ordshuf", ID_SHUFFLE_456, "shuffle_seed456"),
    ):
        name = f"v3h_proto_opsd_{tag}_instrdialog_seed456"
        rows.append(
            (
                "e10",
                proto_backbone(
                    stream="instrdialog",
                    run_name=name,
                    experiment_name=f"v3h_proto_opsd_{tag}_instrdialog",
                    neighbor_ratio=0.1,
                    forgotten_ratio=0.1,
                    replay_ratio=0.2,
                    lambda_opsd=0.2,
                    buffer_k=10,
                    holdout=True,
                    segment_order=order,
                    segment_order_name=order_name,
                ),
            )
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for group, cfg in specs():
        path = dump(cfg)
        written.append((group, path.name))
        print(f"{group}\t{path.name}")
    print(f"wrote {len(written)} configs")


if __name__ == "__main__":
    main()
