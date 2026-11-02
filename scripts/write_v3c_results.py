#!/usr/bin/env python3
"""Build the v3c results note and figures from final_metrics.json only.

Runs without final_metrics.json stay as em dashes. No Protocol A or v3h
numbers are mixed into these means.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3c_matrix import CONTENT, all_runs, parse_name  # noqa: E402

RUNS = Path("/root/autodl-tmp/rp_lora_v3_runs/runs")
LOGDIR = Path("/root/autodl-tmp/rp_lora_v3_runs/logs")
DOC = ROOT / "docs" / "ACL_v3_Corrected_Method_Results.md"
FIGDIR = ROOT / "results" / "figures"
DASH = "—"
KEYS = (
    "seen_avg_score",
    "seen_avg_task_aware_score",
    "forgetting",
    "task_aware_forgetting",
    "token_f1_mean",
    "rouge_l_mean",
    "bleu_mean",
    "lcs_overlap_mean",
    "current_score",
    "current_task_aware_score",
    "task_aware_score_mean",
)
HEADER = (
    "| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | "
    "Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | Current-Seg Acc↑ | "
    "Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |"
)
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
DATA_LABEL = {"instrdialog": "InstrDialog", "instrdialogpp": "InstrDialog++"}


def fmt(value) -> str:
    if value is None:
        return DASH
    return f"{float(value):.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def load_metrics(name: str):
    path = RUNS / name / "final_metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload.get("final") or {}
    out = {}
    for key in KEYS:
        raw = final.get(f"eval.{key}")
        if not isinstance(raw, (int, float)):
            return None
        out[key] = float(raw)
    return out


def resume_cmd(name: str) -> str:
    path = LOGDIR / f"{name}.resume_cmd"
    if path.is_file():
        return str(path)
    return (
        "cd /root/research-repos/opd-lora_v3 && "
        "PYTHONPATH=/root/research-repos/opd-lora_v3 "
        "/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python scripts/run_v3c_queue.py --phase all"
    )


def cell_name(letter: str, objective: str, stream: str, seed: int) -> str:
    return f"v3c_{letter}_{objective}_{stream}_seed{seed}"


def explain(name: str, info: dict, metrics: dict, table: dict) -> str:
    acc = metrics["seen_avg_score"]
    forg = metrics["forgetting"]
    seed = int(info["seed"])
    letter = str(info["letter"])
    objective = str(info["objective"])
    stream = str(info["stream"])
    bits = [f"seed {seed}同协议", f"Acc {acc:.3f},F {forg:.3f}", f"run_id {name}"]
    worse = acc < 0.05
    soft_worse = False

    def pair(other_letter: str, label: str) -> None:
        nonlocal soft_worse
        other = cell_name(other_letter, objective, stream, seed)
        other_m = table.get(other)
        if not other_m:
            bits.append(f"对照 {label} 未完成")
            return
        d_acc = acc - other_m["seen_avg_score"]
        d_f = forg - other_m["forgetting"]
        bits.append(
            f"对照 {label} Acc {other_m['seen_avg_score']:.3f},F {other_m['forgetting']:.3f}"
            f"(ΔAcc {signed(d_acc)},ΔF {signed(d_f)})"
        )
        if letter == "P" and other_letter == "U" and (d_acc <= -0.01 or d_f >= 0.01):
            soft_worse = True
        if letter == "U" and other_letter == "R" and (d_acc <= -0.01 or d_f >= 0.01):
            soft_worse = True

    if letter == "P":
        pair("U", "U")
        pair("R", "R")
    elif letter == "U":
        pair("R", "R")
        pair("P", "P")
    else:
        pair("P", "P")
        pair("U", "U")
    ce_name = cell_name(letter, "ce", stream, seed)
    opsd_name = cell_name(letter, "opsd", stream, seed)
    if objective == "opsd" and table.get(ce_name):
        base = table[ce_name]
        d_acc = acc - base["seen_avg_score"]
        d_f = forg - base["forgetting"]
        bits.append(f"Δ_OPSD Acc {signed(d_acc)},F {signed(d_f)}")
        if d_acc <= -0.01 and d_f >= 0.01:
            soft_worse = True
    elif objective == "ce" and table.get(opsd_name):
        base = table[opsd_name]
        bits.append(
            "Δ_OPSD "
            f"Acc {signed(base['seen_avg_score'] - acc)},F {signed(base['forgetting'] - forg)}"
        )
    if acc < 0.05:
        prefix = "异常:"
    elif worse or soft_worse:
        prefix = "负向:"
    else:
        prefix = "合理:"
    return prefix + ";".join(bits)


def content_for(info: dict) -> str:
    text = CONTENT[str(info["letter"])].format(seed=int(info["seed"]))
    objective = "CE(λ=0)" if info["objective"] == "ce" else "CE+λ=0.2 OPSD"
    return text.replace(", seed", f",{objective}, seed")


def ordered_names():
    names = []
    for stream in ("instrdialog", "instrdialogpp"):
        for letter in ("R", "U", "P"):
            for objective in ("ce", "opsd"):
                for seed in (456, 789, 42):
                    names.append(cell_name(letter, objective, stream, seed))
    return names


def row_for(name: str, table: dict) -> str:
    info = parse_name(name)
    metrics = table.get(name)
    data = DATA_LABEL[str(info["stream"])]
    content = content_for(info)
    if metrics is None:
        cells = [DASH] * 11
        note = f"异常:无 final_metrics.json;run_id {name};续跑 {resume_cmd(name)}"
    else:
        cells = [fmt(metrics[key]) for key in KEYS]
        note = explain(name, info, metrics, table)
    return "| " + " | ".join(["v3c_matched_budget", content, data, *cells, note]) + " |"


def mean_std(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def delta_lines(table: dict) -> list:
    lines = [
        "## 配对效应",
        "",
        "只在 v3c 已完成的同 seed、同数据、同目标格子上做差。遗忘下降是改善。",
        "Δ_prototype = P−U，Δ_PGDR = P−R，Δ_OPSD = OPSD−CE。",
        "交互项 = (P 的 Δ_OPSD) − (R 的 Δ_OPSD)。缺任意一格就写 —，不补均值。",
        "",
        "| Data | Objective | Seed | Δ_prototype Acc | Δ_prototype F | Δ_PGDR Acc | Δ_PGDR F | Δ_OPSD(P) Acc | Δ_OPSD(P) F | 交互 Acc | 交互 F |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for stream in ("instrdialog", "instrdialogpp"):
        for objective in ("ce", "opsd"):
            for seed in (456, 789, 42):
                p = table.get(cell_name("P", objective, stream, seed))
                u = table.get(cell_name("U", objective, stream, seed))
                r = table.get(cell_name("R", objective, stream, seed))
                p_ce = table.get(cell_name("P", "ce", stream, seed))
                p_opsd = table.get(cell_name("P", "opsd", stream, seed))
                r_ce = table.get(cell_name("R", "ce", stream, seed))
                r_opsd = table.get(cell_name("R", "opsd", stream, seed))

                def diff(left, right, key):
                    if not left or not right:
                        return DASH
                    return signed(left[key] - right[key])

                d_opsd_p_acc = diff(p_opsd, p_ce, "seen_avg_score")
                d_opsd_p_f = diff(p_opsd, p_ce, "forgetting")
                if p_opsd and p_ce and r_opsd and r_ce:
                    inter_acc = signed(
                        (p_opsd["seen_avg_score"] - p_ce["seen_avg_score"])
                        - (r_opsd["seen_avg_score"] - r_ce["seen_avg_score"])
                    )
                    inter_f = signed(
                        (p_opsd["forgetting"] - p_ce["forgetting"])
                        - (r_opsd["forgetting"] - r_ce["forgetting"])
                    )
                else:
                    inter_acc = inter_f = DASH
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            DATA_LABEL[stream],
                            objective,
                            str(seed),
                            diff(p, u, "seen_avg_score"),
                            diff(p, u, "forgetting"),
                            diff(p, r, "seen_avg_score"),
                            diff(p, r, "forgetting"),
                            d_opsd_p_acc,
                            d_opsd_p_f,
                            inter_acc,
                            inter_f,
                        ]
                    )
                    + " |"
                )
    lines.append("")
    lines.append("v3c 三种子均值只在三格都有 final_metrics.json 时计算，不与 v3h 或 Protocol A 混合。")
    lines.append("")
    lines.append("| Data | Letter | Objective | n | Acc mean | Acc std | F mean | F std |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for stream in ("instrdialog", "instrdialogpp"):
        for letter in ("R", "U", "P"):
            for objective in ("ce", "opsd"):
                accs, fs = [], []
                for seed in (456, 789, 42):
                    row = table.get(cell_name(letter, objective, stream, seed))
                    if row:
                        accs.append(row["seen_avg_score"])
                        fs.append(row["forgetting"])
                if len(accs) < 3:
                    lines.append(
                        f"| {DATA_LABEL[stream]} | {letter} | {objective} | {len(accs)} | {DASH} | {DASH} | {DASH} | {DASH} |"
                    )
                    continue
                acc_mean, acc_std = mean_std(accs)
                f_mean, f_std = mean_std(fs)
                lines.append(
                    f"| {DATA_LABEL[stream]} | {letter} | {objective} | 3 | {acc_mean:.3f} | {acc_std:.3f} | {f_mean:.3f} | {f_std:.3f} |"
                )
    return lines


def gpu_seconds() -> float:
    path = LOGDIR / "v3c_queue.steps.tsv"
    if not path.is_file():
        return 0.0
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 4 or parts[2] != "OK":
            continue
        extra = parts[3]
        if extra.startswith("sec="):
            total += float(extra.split("=", 1)[1])
    return total


def write_doc(table: dict) -> None:
    names = ordered_names()
    done = sum(1 for name in names if table.get(name))
    missing = [name for name in names if not table.get(name)]
    lines = [
        "# v3c 修正配额与真正原型检索",
        "",
        "本表只含 `v3c_matched_budget`。mixed-batch ρ=0.20 在 45 条 current 上的目标是 11 条回放，实现比例 11/56，不是严格 20%。",
        "R 是全历史均匀，U 是格式兼容后的均匀 Related 加 train-holdout 退化，P 是显式 L2 余弦。U 不是 cosine prototype。",
        "旧 `v3h_*` 的 9 对 8 不进入本表，也不与这里的三种子一起算 Mean±Std。Protocol A 同样分开。",
        f"完成 {done}/36。GPU 时间（成功 run 的 elapsed 之和）{gpu_seconds():.0f} 秒。",
        "",
        "```md",
        HEADER,
        SEP,
    ]
    for name in names:
        lines.append(row_for(name, table))
    lines.append("```")
    lines.append("")
    lines.extend(delta_lines(table))
    lines.append("")
    if missing:
        lines.append("## 未完成")
        lines.append("")
        for name in missing:
            lines.append(f"- {name}: 无 final_metrics.json。续跑 `{resume_cmd(name)}`")
        lines.append("")
    lines.append("## 图")
    lines.append("")
    lines.append("图只使用已有 `final_metrics.json` 或 `replay_log.jsonl`。没有实测的序列不画。")
    lines.append("")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _measured_points(table: dict):
    points = []
    for name, metrics in table.items():
        info = parse_name(name)
        points.append((info, metrics))
    return points


def write_figures(table: dict) -> list:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"NO_FIGURES {exc}")
        return []
    FIGDIR.mkdir(parents=True, exist_ok=True)
    written = []
    points = _measured_points(table)
    if not points:
        print("NO_MEASURED_POINTS")
        return []

    def save(fig, name: str) -> None:
        path = FIGDIR / name
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    # prototype vs uniform: only seeds where both P and U exist
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, key, title in (
        (axes[0], "seen_avg_score", "Seen-Avg Acc"),
        (axes[1], "forgetting", "Forgetting"),
    ):
        xs, ys = [], []
        for stream in ("instrdialog", "instrdialogpp"):
            for objective in ("ce", "opsd"):
                for seed in (456, 789, 42):
                    p = table.get(cell_name("P", objective, stream, seed))
                    u = table.get(cell_name("U", objective, stream, seed))
                    if not p or not u:
                        continue
                    xs.append(u[key])
                    ys.append(p[key])
        if xs:
            lo = min(xs + ys)
            hi = max(xs + ys)
            pad = max(0.01, (hi - lo) * 0.1)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.7", linewidth=1)
            ax.scatter(xs, ys, s=28)
        ax.set_xlabel("U")
        ax.set_ylabel("P")
        ax.set_title(title)
    save(fig, "prototype_vs_uniform.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    labels, accs, forgets = [], [], []
    for stream in ("instrdialog", "instrdialogpp"):
        for objective in ("ce", "opsd"):
            for letter in ("R", "U", "P"):
                vals_a, vals_f = [], []
                for seed in (456, 789, 42):
                    row = table.get(cell_name(letter, objective, stream, seed))
                    if row:
                        vals_a.append(row["seen_avg_score"])
                        vals_f.append(row["forgetting"])
                if not vals_a:
                    continue
                labels.append(f"{letter}-{objective}\n{DATA_LABEL[stream][:2]}")
                accs.append(sum(vals_a) / len(vals_a))
                forgets.append(sum(vals_f) / len(vals_f))
    if labels:
        axes[0].bar(range(len(labels)), accs)
        axes[1].bar(range(len(labels)), forgets)
        for ax, title in ((axes[0], "Acc"), (axes[1], "Forgetting")):
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(title)
    save(fig, "budget_matched_comparison.pdf")

    def factorial(key: str, filename: str, title: str) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
        for ax, stream in zip(axes, ("instrdialog", "instrdialogpp")):
            for letter, style in (("R", "o-"), ("P", "s-")):
                ys = []
                xs = []
                for i, objective in enumerate(("ce", "opsd")):
                    vals = []
                    for seed in (456, 789, 42):
                        row = table.get(cell_name(letter, objective, stream, seed))
                        if row:
                            vals.append(row[key])
                    if not vals:
                        continue
                    xs.append(i)
                    ys.append(sum(vals) / len(vals))
                if xs:
                    ax.plot(xs, ys, style, label=letter)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["CE", "CE+OPSD"])
            ax.set_title(DATA_LABEL[stream])
            ax.legend(frameon=False)
        fig.suptitle(title)
        save(fig, filename)

    factorial("seen_avg_score", "factorial_accuracy.pdf", "Factorial accuracy")
    factorial("forgetting", "factorial_forgetting.pdf", "Factorial forgetting")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    plotted = False
    for letter, color in (("R", "#4c78a8"), ("U", "#f58518"), ("P", "#54a24b")):
        scores = []
        for stream in ("instrdialog", "instrdialogpp"):
            for objective in ("ce", "opsd"):
                for seed in (456, 789, 42):
                    name = cell_name(letter, objective, stream, seed)
                    path = RUNS / name / "replay_log.jsonl"
                    if not path.is_file():
                        continue
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        for value in row.get("selected_cosine_scores") or []:
                            if isinstance(value, (int, float)):
                                scores.append(float(value))
        if scores:
            ax.hist(scores, bins=20, alpha=0.45, label=letter, color=color)
            plotted = True
    if plotted:
        ax.set_xlabel("Selected cosine")
        ax.set_ylabel("Count")
        ax.legend(frameon=False)
    save(fig, "selected_cosine_distribution.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, key, title in (
        (axes[0], "seen_avg_score", "Δ Acc"),
        (axes[1], "forgetting", "Δ Forgetting"),
    ):
        for stream, marker in (("instrdialog", "o"), ("instrdialogpp", "s")):
            for objective, color in (("ce", "#4c78a8"), ("opsd", "#e45756")):
                xs, ys = [], []
                for seed in (456, 789, 42):
                    p = table.get(cell_name("P", objective, stream, seed))
                    u = table.get(cell_name("U", objective, stream, seed))
                    if not p or not u:
                        continue
                    xs.append(seed)
                    ys.append(p[key] - u[key])
                if xs:
                    ax.scatter(xs, ys, marker=marker, color=color, label=f"{DATA_LABEL[stream]} {objective}")
        ax.axhline(0.0, color="0.6", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("seed")
    axes[0].legend(fontsize=7, frameon=False)
    save(fig, "per_seed_effects.pdf")
    return written


def main() -> None:
    table = {}
    for name in all_runs():
        metrics = load_metrics(name)
        if metrics is not None:
            table[name] = metrics
    write_doc(table)
    figures = write_figures(table)
    print(f"DOC {DOC}")
    print(f"COMPLETED {len(table)}/{len(all_runs())}")
    print(f"FIGURES {len(figures)}")
    for path in figures:
        print(path)


if __name__ == "__main__":
    main()
