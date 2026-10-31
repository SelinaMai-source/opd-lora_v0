#!/usr/bin/env python3
"""Append pending-extension rows from final_metrics.json only. Missing = em dash."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ROOT = Path("/root/research-repos/opd-lora_v3")
RUNS = Path("/root/autodl-tmp/rp_lora_v3_runs/runs")
LOGS = Path("/root/autodl-tmp/rp_lora_v3_runs/logs")
DOCS = ROOT / "docs"
DASH = "—"

KEYS = [
    ("seen_avg_score", "Seen-Avg Acc"),
    ("seen_avg_task_aware_score", "Seen-Avg Task-aware Acc"),
    ("forgetting", "Forgetting"),
    ("task_aware_forgetting", "Task-Aware Forgetting"),
    ("token_f1_mean", "Token F1"),
    ("rouge_l_mean", "ROUGE-L"),
    ("bleu_mean", "BLEU"),
    ("lcs_overlap_mean", "LCS Overlap"),
    ("current_score", "Current-Seg Acc"),
    ("current_task_aware_score", "Current-Seg Task-aware Acc"),
    ("task_aware_score_mean", "Task-aware Score Mean"),
]

GROUPS = {
    "e2": [
        "v3h_proto_opsd_n15f5_instrdialog_seed456",
        "v3h_proto_opsd_n5f15_instrdialog_seed456",
        "v3h_proto_opsd_n15f5_instrdialogpp_seed456",
        "v3h_proto_opsd_n5f15_instrdialogpp_seed456",
    ],
    "e4lam": [
        "v3h_proto_opsd_lam005_instrdialog_seed456",
        "v3h_proto_opsd_lam010_instrdialog_seed456",
        "v3h_proto_opsd_lam050_instrdialog_seed456",
        "v3h_proto_opsd_lam005_instrdialogpp_seed456",
        "v3h_proto_opsd_lam010_instrdialogpp_seed456",
        "v3h_proto_opsd_lam050_instrdialogpp_seed456",
    ],
    "e4off": [
        "v3h_proto_opsd_offpol_instrdialog_seed456",
        "v3h_proto_opsd_offpol_instrdialogpp_seed456",
    ],
    "e6": [
        "v3_proto_opsd_rr0_instrdialog_seed456",
        "v3h_proto_opsd_rr10_instrdialog_seed456",
        "v3h_proto_opsd_rr30_instrdialog_seed456",
        "v3h_proto_opsd_rr40_instrdialog_seed456",
        "v3_proto_opsd_rr0_instrdialogpp_seed456",
        "v3h_proto_opsd_rr10_instrdialogpp_seed456",
        "v3h_proto_opsd_rr30_instrdialogpp_seed456",
        "v3h_proto_opsd_rr40_instrdialogpp_seed456",
    ],
    "e3": [
        "v3_opsd_direct_instrdialog_seed456",
        "v3_opsd_direct_instrdialogpp_seed456",
        "v3_seg_opsd_k25_instrdialog_seed456",
        "v3_seg_opsd_k25_instrdialogpp_seed456",
    ],
    "e9": [
        "v3h_proto_opsd_bufk5_instrdialog_seed456",
        "v3h_proto_opsd_bufk20_instrdialog_seed456",
        "v3h_proto_opsd_bufk50_instrdialog_seed456",
        "v3h_proto_opsd_bufk5_instrdialogpp_seed456",
        "v3h_proto_opsd_bufk20_instrdialogpp_seed456",
        "v3h_proto_opsd_bufk50_instrdialogpp_seed456",
    ],
    "e10": [
        "v3h_proto_opsd_ordrev_instrdialog_seed456",
        "v3h_proto_opsd_ordshuf_instrdialog_seed456",
    ],
    "e11": [
        "v3h_proto_opsd_allnb_instrdialog_seed456",
        "v3h_proto_opsd_allfg_instrdialog_seed456",
        "v3h_proto_opsd_allnb_instrdialogpp_seed456",
        "v3h_proto_opsd_allfg_instrdialogpp_seed456",
    ],
}

TITLES = {
    "e2": "E2 15/5 与 5/15 槽混合（train_holdout）",
    "e4lam": "E4 λ ∈ {0.05, 0.10, 0.50}（PGDR+CE+OPSD holdout）",
    "e4off": "E4 off-policy gold tokens vs on-policy（λ=0.2 holdout）",
    "e6": "E6 replay_ratio ∈ {0, 0.10, 0.30, 0.40}",
    "e3": "E3 v3 协议对照（direct OPSD / seg_opsd K=25）",
    "e9": "E9 buffer_k ∈ {5, 20, 50}",
    "e10": "E10 InstrDialog 任务顺序置换",
    "e11": "E11 holdout 全相关 / 全退化 20%",
}


def fmt(value: Optional[float]) -> str:
    if value is None:
        return DASH
    return f"{value:.3f}"


def load_eleven(name: str) -> dict:
    path = RUNS / name / "final_metrics.json"
    out = {key: None for key, _ in KEYS}
    out["_path"] = None
    if not path.is_file():
        return out
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload.get("final") or {}
    for key, _ in KEYS:
        raw = final.get(f"eval.{key}")
        out[key] = float(raw) if isinstance(raw, (int, float)) else None
    out["_path"] = str(path)
    return out


def eleven_table(names: list[str]) -> str:
    headers = ["run"] + [label for _, label in KEYS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name in names:
        metrics = load_eleven(name)
        row = [name] + [fmt(metrics[key]) for key, _ in KEYS]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def timing_row(name: str) -> str:
    start = (LOGS / f"{name}.start").read_text(encoding="utf-8").strip() if (LOGS / f"{name}.start").is_file() else DASH
    end = (LOGS / f"{name}.end").read_text(encoding="utf-8").strip() if (LOGS / f"{name}.end").is_file() else DASH
    elapsed = (LOGS / f"{name}.elapsed_sec").read_text(encoding="utf-8").strip() if (LOGS / f"{name}.elapsed_sec").is_file() else DASH
    gpu = (LOGS / f"{name}.gpu_after").read_text(encoding="utf-8").strip() if (LOGS / f"{name}.gpu_after").is_file() else DASH
    return f"| {name} | {start} | {end} | {elapsed} | {gpu} |"


def replace_or_append_section(text: str, marker: str, body: str) -> str:
    start = f"<!-- BEGIN {marker} -->"
    end = f"<!-- END {marker} -->"
    block = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        return before + block + after
    return text.rstrip() + "\n\n" + block + "\n"


def pending_body() -> str:
    chunks = [
        "## Pending extensions（seed 456，本轮新跑）",
        "",
        "数字只来自 `final_metrics.json`。缺测写 —。JSD 无单独蒸馏温度；`rollout_temperature` 保持 1.0。",
        "E3 SFT→OPSD：Blocked（CITB Sequential adapter 走非 multiref 流且快照无 format_normalize，不是合法 v3 协议 init）。",
        "不重跑 20% 随机 / 全相关 / 全退化 / 10+10 历史格，也不开 50–90% replay 或新的 789/42。",
        "",
    ]
    for group, title in TITLES.items():
        chunks.append(f"### {title}")
        chunks.append("")
        chunks.append(eleven_table(GROUPS[group]))
        chunks.append("")
        chunks.append("| run | start | end | elapsed_sec | gpu_after |")
        chunks.append("|---|---|---|---|---|")
        for name in GROUPS[group]:
            chunks.append(timing_row(name))
        chunks.append("")
    return "\n".join(chunks)


def queue_body() -> str:
    lines = [
        "# ABLATION v3 Run Queue",
        "",
        "单卡顺序。已有 `final_metrics.json` 且配置指纹一致则跳过。目录已存在但不完整则拒绝覆盖，续跑命令在 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。",
        "",
        "结果目录：`/root/autodl-tmp/rp_lora_v3_runs/runs`。",
        "",
        "## 已完成矩阵（不重跑）",
        "",
        "- 冒烟 `v3h_proto_opsd_instrdialog_seed456_smoke2seg`",
        "- 旧协议 12（seed 789/42 的 rand_ce / proto_ce / rand_opsd × 两流）",
        "- v3h 8（seed 456 train_holdout 2×2×两流）",
        "- 历史 20% 随机 / 全相关 / 全退化 / 10+10、CITB / LCIA / v0 / v1 / v2 / Lit",
        "",
        "## Pending extensions 顺序",
        "",
        "E2 → E4 λ → E4 off-policy → E6 → E3 → E9 → E10 → E11。`forgotten_ratio>0` 的新 run 用 `train_holdout`，run_name 带 `v3h_`。",
        "",
    ]
    for group, title in TITLES.items():
        lines.append(f"### {title}")
        for name in GROUPS[group]:
            metrics = load_eleven(name)
            status = "done" if metrics.get("seen_avg_score") is not None else "pending"
            lines.append(f"- `{name}`: {status}")
        lines.append("")
    lines.extend(
        [
            "### Blocked",
            "",
            "- E3 SFT→OPSD：无合法 init adapter（Sequential 快照为非 `*_multiref.json` 且无 `format_normalize`）。",
            "- 蒸馏温度扫描：不做。原目标 JSD 无单独 T，`rollout_temperature` 为 1.0。",
            "",
            "## 不做",
            "",
            "不重跑 CITB / LCIA / v0 / v1 / v2 / Lit。不开 50–90% replay。不开新的 seed 789/42。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    results = DOCS / "ABLATION_V3_实验结果与待补表.md"
    queue = DOCS / "ABLATION_V3_Run_Queue.md"
    text = results.read_text(encoding="utf-8") if results.exists() else ""
    # Drop the old "本轮不跑" pending stub if still present.
    if "## 待补（本轮不跑）" in text:
        text = text.split("## 待补（本轮不跑）")[0].rstrip() + "\n"
    results.write_text(replace_or_append_section(text, "PENDING_EXT", pending_body()), encoding="utf-8")
    queue.write_text(queue_body(), encoding="utf-8")
    print("updated docs")


if __name__ == "__main__":
    main()
