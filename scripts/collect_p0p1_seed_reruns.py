#!/usr/bin/env python3
"""Copy metrics and write experiments_v1/seed_reruns.md (numbers only)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/rp_lora_v0")
RUNS = ROOT / "results" / "runs"
TABLES = ROOT / "results" / "tables"

KEYS = [
    ("seen_avg", "eval.seen_avg_score"),
    ("F", "eval.forgetting"),
    ("token_f1", "eval.token_f1_mean"),
    ("rouge_l", "eval.rouge_l_mean"),
    ("bleu", "eval.bleu_mean"),
]

# method, data, seed_label, seed, run_name, dest_dir, dest_stem
ROWS = [
    ("v1_teacher_enhance_OPSD", "InstrDialog", "seed 123 重跑", 123,
     "v1_teacher_enhance_opsd_instrdialog_seed123_rerun",
     "experiments_v1/v1_teacher_enhance_OPSD/results", "instrdialog_seed123_rerun"),
    ("v1_teacher_enhance_OPSD", "InstrDialog", "seed 456", 456,
     "v1_teacher_enhance_opsd_instrdialog_seed456",
     "experiments_v1/v1_teacher_enhance_OPSD/results", "instrdialog_seed456"),
    ("v1_teacher_enhance_OPSD", "InstrDialog++", "seed 123 重跑", 123,
     "v1_teacher_enhance_opsd_instrdialogpp_seed123_rerun",
     "experiments_v1/v1_teacher_enhance_OPSD/results", "instrdialogpp_seed123_rerun"),
    ("v1_teacher_enhance_OPSD", "InstrDialog++", "seed 456", 456,
     "v1_teacher_enhance_opsd_instrdialogpp_seed456",
     "experiments_v1/v1_teacher_enhance_OPSD/results", "instrdialogpp_seed456"),
    ("v1_seg_OPSD_replay K=25", "InstrDialog", "seed 123 重跑", 123,
     "v1_seg_opsd_replay_instrdialog_seed123_rerun",
     "experiments_v1/v1_seg_OPSD_replay/results", "instrdialog_seed123_rerun"),
    ("v1_seg_OPSD_replay K=25", "InstrDialog", "seed 456", 456,
     "v1_seg_opsd_replay_instrdialog_seed456",
     "experiments_v1/v1_seg_OPSD_replay/results", "instrdialog_seed456"),
    ("v1_seg_OPSD_replay K=25", "InstrDialog++", "seed 123 重跑", 123,
     "v1_seg_opsd_replay_instrdialogpp_seed123_rerun",
     "experiments_v1/v1_seg_OPSD_replay/results", "instrdialogpp_seed123_rerun"),
    ("v1_seg_OPSD_replay K=25", "InstrDialog++", "seed 456", 456,
     "v1_seg_opsd_replay_instrdialogpp_seed456",
     "experiments_v1/v1_seg_OPSD_replay/results", "instrdialogpp_seed456"),
    ("v1_seg_OPSD K=25", "InstrDialog", "seed 123 重跑", 123,
     "v1_seg_opsd_k25_instrdialog_seed123_rerun",
     "experiments_v1/v1_seg_OPSD/results", "instrdialog_k25_seed123_rerun"),
    ("v1_seg_OPSD K=25", "InstrDialog", "seed 456", 456,
     "v1_seg_opsd_k25_instrdialog_seed456",
     "experiments_v1/v1_seg_OPSD/results", "instrdialog_k25_seed456"),
    ("v1_seg_OPSD K=25", "InstrDialog++", "seed 123 重跑", 123,
     "v1_seg_opsd_k25_instrdialogpp_seed123_rerun",
     "experiments_v1/v1_seg_OPSD/results", "instrdialogpp_k25_seed123_rerun"),
    ("v1_seg_OPSD K=25", "InstrDialog++", "seed 456", 456,
     "v1_seg_opsd_k25_instrdialogpp_seed456",
     "experiments_v1/v1_seg_OPSD/results", "instrdialogpp_k25_seed456"),
    ("v1_SFT_OPSD", "InstrDialog++", "seed 123 重跑", 123,
     "v1_sft_opsd_instrdialogpp_seed123_rerun",
     "experiments_v1/v1_SFT_OPSD/results", "instrdialogpp_seed123_rerun"),
    ("v1_SFT_OPSD", "InstrDialog++", "seed 456", 456,
     "v1_sft_opsd_instrdialogpp_seed456",
     "experiments_v1/v1_SFT_OPSD/results", "instrdialogpp_seed456"),
]


def _fmt(x) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def main() -> int:
    lines = [
        "# P0+P1 双 seed 重跑清单",
        "",
        "共 14 次：teacher_enhance / seg_OPSD_replay / seg_OPSD K=25 / SFT_OPSD++。",
        "原 run 未覆盖（固定 `run_name`）。SFT init 变体两边 seed 共用同一份 v0 SFT adapter。",
        "",
        "| # | 方法 | Data | seed | 内容 | run_name | run 目录 | seen_avg↑ | F↓ | token_f1↑ | rouge_l↑ | bleu↑ |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    n_ok = 0
    for i, (method, data, seed_label, seed, run_name, dest_rel, dest_stem) in enumerate(ROWS, 1):
        run_dir = RUNS / run_name
        fm = run_dir / "final_metrics.json"
        dest_dir = ROOT / dest_rel
        dest_dir.mkdir(parents=True, exist_ok=True)
        metrics = {}
        if fm.is_file():
            n_ok += 1
            shutil.copy2(fm, dest_dir / f"final_metrics_{dest_stem}.json")
            csv_src = TABLES / f"{run_name}_segment_metrics.csv"
            if csv_src.is_file():
                shutil.copy2(csv_src, dest_dir / f"segment_metrics_{dest_stem}.csv")
            payload = json.loads(fm.read_text(encoding="utf-8"))
            metrics = payload.get("final") or {}
        vals = [_fmt(metrics.get(k)) for _, k in KEYS]
        run_disp = str(run_dir.relative_to(ROOT)) if fm.is_file() else "MISSING"
        lines.append(
            f"| {i} | {method} | {data} | {seed} | {seed_label} | `{run_name}` | `{run_disp}` | "
            + " | ".join(vals)
            + " |"
        )
    lines.extend(
        [
            "",
            f"完成 {n_ok}/14。指标提取自各 run 的 `final_metrics.json`，保留 3 位小数。",
            "",
        ]
    )
    out = ROOT / "experiments_v1" / "seed_reruns.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ok={n_ok}/14")
    return 0 if n_ok == 14 else 1


if __name__ == "__main__":
    raise SystemExit(main())
