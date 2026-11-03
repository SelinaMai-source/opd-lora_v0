#!/usr/bin/env python3
"""Build the v3c results note from final.eval only, then draw measured figures.

Runs without final_metrics.json stay as em dashes. Protocol A and v3h numbers
are not read and are not mixed into these means.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v3c_matrix import CONTENT, all_runs, parse_name  # noqa: E402

RUNS = Path(os.environ.get("OPD_RUNS_DIR", "/root/autodl-tmp/rp_lora_v3_runs/runs"))
LOGDIR = Path(os.environ.get("OPD_LOG_DIR", "/root/autodl-tmp/rp_lora_v3_runs/logs"))
DOC = ROOT / "docs" / "ACL_v3_Corrected_Method_Results.md"
FIG_SCRIPT = ROOT / "results" / "figures" / "plot_v3c_corrected.py"
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
SEP = "|" + "|".join(["---"] * 15) + "|"
DATA_LABEL = {"instrdialog": "InstrDialog", "instrdialogpp": "InstrDialog++"}
OBJ_LABEL = {"ce": "CE", "opsd": "CE+OPSD"}
SEEDS = (456, 789, 42)


def r3(value: float) -> float:
    return float(f"{float(value):.3f}")


def fmt(value) -> str:
    if value is None:
        return DASH
    return f"{float(value):.3f}"


def signed(value: float) -> str:
    return f"{value:+.3f}"


def mean_std(values):
    if not values:
        return None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def pm(values) -> str:
    stats = mean_std(values)
    if stats is None:
        return DASH
    mean, std = stats
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f}±{std:.3f}"


def load_metrics(name: str):
    path = RUNS / name / "final_metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or "")
    if run_id != name:
        raise SystemExit(f"{name} run_id is {run_id or 'missing'}")
    final = payload.get("final") or {}
    out = {}
    for key in KEYS:
        raw = final.get(f"eval.{key}")
        if not isinstance(raw, (int, float)):
            return None
        out[key] = float(raw)
    return out


def cell_name(letter: str, objective: str, stream: str, seed: int) -> str:
    return f"v3c_{letter}_{objective}_{stream}_seed{seed}"


def explain(name: str, info: dict, metrics: dict, table: dict) -> str:
    acc = r3(metrics["seen_avg_score"])
    forg = r3(metrics["forgetting"])
    seed = int(info["seed"])
    letter = str(info["letter"])
    objective = str(info["objective"])
    stream = str(info["stream"])
    bits = [f"seed {seed}同协议", f"Acc {acc:.3f},F {forg:.3f}", f"run_id {name}"]
    soft_worse = False

    def worse(d_acc: float, d_f: float) -> bool:
        return float(f"{d_acc:.3f}") <= -0.01 or float(f"{d_f:.3f}") >= 0.01

    def pair(other_letter: str, label: str) -> None:
        nonlocal soft_worse
        other = cell_name(other_letter, objective, stream, seed)
        other_m = table.get(other)
        if not other_m:
            bits.append(f"对照 {label} 未完成")
            return
        d_acc = acc - r3(other_m["seen_avg_score"])
        d_f = forg - r3(other_m["forgetting"])
        bits.append(
            f"对照 {label} Acc {r3(other_m['seen_avg_score']):.3f},F {r3(other_m['forgetting']):.3f}"
            f"(本行−对照 ΔAcc {signed(d_acc)},ΔF {signed(d_f)})"
        )
        if letter == "P" and other_letter in {"U", "R"} and worse(d_acc, d_f):
            soft_worse = True
        if letter == "U" and other_letter == "R" and worse(d_acc, d_f):
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
        d_acc = acc - r3(base["seen_avg_score"])
        d_f = forg - r3(base["forgetting"])
        bits.append(f"Δ_OPSD Acc {signed(d_acc)},F {signed(d_f)}")
        if float(f"{d_acc:.3f}") <= -0.01 and float(f"{d_f:.3f}") >= 0.01:
            soft_worse = True
    elif objective == "ce" and table.get(opsd_name):
        base = table[opsd_name]
        bits.append(
            "Δ_OPSD "
            f"Acc {signed(r3(base['seen_avg_score']) - acc)},F {signed(r3(base['forgetting']) - forg)}"
        )
    if acc < 0.05:
        prefix = "异常:"
    elif soft_worse:
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
                for seed in SEEDS:
                    names.append(cell_name(letter, objective, stream, seed))
    return names


def row_for(name: str, table: dict) -> str:
    info = parse_name(name)
    metrics = table.get(name)
    data = DATA_LABEL[str(info["stream"])]
    content = content_for(info)
    if metrics is None:
        cells = [DASH] * 11
        note = f"异常:无 final_metrics.json;run_id {name}"
    else:
        cells = [fmt(metrics[key]) for key in KEYS]
        note = explain(name, info, metrics, table)
    return "| " + " | ".join(["v3c_matched_budget", content, data, *cells, note]) + " |"


def _get(table, letter, objective, stream, seed):
    return table.get(cell_name(letter, objective, stream, seed))


def delta_lines(table: dict) -> list:
    lines = [
        "## 配对效应",
        "",
        "只在 v3c 同 seed、同数据、同目标的格子上做差。差由上表三位小数相减。",
        "遗忘下降才是改善：ΔF 为负才算遗忘变好。P 低于 U 或 R 的差保留负号。",
        "Δ_prototype = P−U，Δ_PGDR = P−R，Δ_OPSD = OPSD−CE。",
        "交互项 I = (P 的 Δ_OPSD) − (R 的 Δ_OPSD)。I 只取决于数据和 seed，同一 seed 的 CE 行与 CE+OPSD 行写同一个 I。",
        "Protocol A 和 v3h 的 9 对 8 不进入这些差，也不进入后面的 Mean±Std。",
        "",
        "| Data | Objective | Seed | Δ_prototype Acc | Δ_prototype F | Δ_PGDR Acc | Δ_PGDR F | Δ_OPSD(P) Acc | Δ_OPSD(P) F | I Acc | I F |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for stream in ("instrdialog", "instrdialogpp"):
        for objective in ("ce", "opsd"):
            for seed in SEEDS:
                p = _get(table, "P", objective, stream, seed)
                u = _get(table, "U", objective, stream, seed)
                r = _get(table, "R", objective, stream, seed)
                p_ce = _get(table, "P", "ce", stream, seed)
                p_opsd = _get(table, "P", "opsd", stream, seed)
                r_ce = _get(table, "R", "ce", stream, seed)
                r_opsd = _get(table, "R", "opsd", stream, seed)

                def diff(left, right, key):
                    if not left or not right:
                        return DASH
                    return signed(r3(left[key]) - r3(right[key]))

                if p_opsd and p_ce and r_opsd and r_ce:
                    inter_acc = signed(
                        (r3(p_opsd["seen_avg_score"]) - r3(p_ce["seen_avg_score"]))
                        - (r3(r_opsd["seen_avg_score"]) - r3(r_ce["seen_avg_score"]))
                    )
                    inter_f = signed(
                        (r3(p_opsd["forgetting"]) - r3(p_ce["forgetting"]))
                        - (r3(r_opsd["forgetting"]) - r3(r_ce["forgetting"]))
                    )
                else:
                    inter_acc = inter_f = DASH
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            DATA_LABEL[stream],
                            OBJ_LABEL[objective],
                            str(seed),
                            diff(p, u, "seen_avg_score"),
                            diff(p, u, "forgetting"),
                            diff(p, r, "seen_avg_score"),
                            diff(p, r, "forgetting"),
                            diff(p_opsd, p_ce, "seen_avg_score"),
                            diff(p_opsd, p_ce, "forgetting"),
                            inter_acc,
                            inter_f,
                        ]
                    )
                    + " |"
                )
    return lines


def _raw_triplet(table, letter, objective, stream, key):
    values = []
    for seed in SEEDS:
        row = _get(table, letter, objective, stream, seed)
        if not row:
            return None
        values.append(float(row[key]))
    return values


def mean_lines(table: dict) -> list:
    lines = [
        "## 三种子 Mean±样本标准差",
        "",
        "每一格都要有 seed 456、789、42 的 `final.eval` 才计算。样本标准差分母是 n−1。",
        "均值和标准差用未四舍五入的原值计算，再保留三位。只含 v3c_matched_budget。",
        "",
        "| Data | Letter | Objective | n | Acc mean | Acc std | F mean | F std |",
        "|---|---|---|---|---|---|---|---|",
    ]
    summaries = []
    for stream in ("instrdialog", "instrdialogpp"):
        for letter in ("R", "U", "P"):
            for objective in ("ce", "opsd"):
                accs = _raw_triplet(table, letter, objective, stream, "seen_avg_score")
                fs = _raw_triplet(table, letter, objective, stream, "forgetting")
                if not accs or not fs:
                    n = 0 if not accs else len(accs)
                    lines.append(
                        f"| {DATA_LABEL[stream]} | {letter} | {OBJ_LABEL[objective]} | {n} | {DASH} | {DASH} | {DASH} | {DASH} |"
                    )
                    continue
                acc_mean, acc_std = mean_std(accs)
                f_mean, f_std = mean_std(fs)
                lines.append(
                    f"| {DATA_LABEL[stream]} | {letter} | {OBJ_LABEL[objective]} | 3 | {acc_mean:.3f} | {acc_std:.3f} | {f_mean:.3f} | {f_std:.3f} |"
                )
                summaries.append((stream, letter, objective, accs, fs, acc_mean, acc_std, f_mean, f_std))
    return lines, summaries


def _delta_triplet(table, stream, objective, left, right, key):
    values = []
    for seed in SEEDS:
        a = _get(table, left, objective, stream, seed)
        b = _get(table, right, objective, stream, seed)
        if not a or not b:
            return None
        values.append(r3(a[key]) - r3(b[key]))
    return values


def _opsd_triplet(table, stream, letter, key):
    values = []
    for seed in SEEDS:
        opsd = _get(table, letter, "opsd", stream, seed)
        ce = _get(table, letter, "ce", stream, seed)
        if not opsd or not ce:
            return None
        values.append(r3(opsd[key]) - r3(ce[key]))
    return values


def effect_mean_lines(table: dict) -> list:
    lines = [
        "## 效应的三种子均值",
        "",
        "这里的差先按上表三位小数相减，再对三颗 seed 取均值和样本标准差。",
        "ΔF 为负才是遗忘改善。",
        "",
        "| Data | Objective | Δ_prototype Acc | Δ_prototype F | Δ_PGDR Acc | Δ_PGDR F |",
        "|---|---|---|---|---|---|",
    ]
    stored = []
    for stream in ("instrdialog", "instrdialogpp"):
        for objective in ("ce", "opsd"):
            dpu_a = _delta_triplet(table, stream, objective, "P", "U", "seen_avg_score")
            dpu_f = _delta_triplet(table, stream, objective, "P", "U", "forgetting")
            dpr_a = _delta_triplet(table, stream, objective, "P", "R", "seen_avg_score")
            dpr_f = _delta_triplet(table, stream, objective, "P", "R", "forgetting")
            if not all((dpu_a, dpu_f, dpr_a, dpr_f)):
                lines.append(
                    f"| {DATA_LABEL[stream]} | {OBJ_LABEL[objective]} | {DASH} | {DASH} | {DASH} | {DASH} |"
                )
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        DATA_LABEL[stream],
                        OBJ_LABEL[objective],
                        pm(dpu_a),
                        pm(dpu_f),
                        pm(dpr_a),
                        pm(dpr_f),
                    ]
                )
                + " |"
            )
            stored.append((stream, objective, dpu_a, dpu_f, dpr_a, dpr_f))
    lines.extend(
        [
            "",
            "| Data | Letter | Δ_OPSD Acc | Δ_OPSD F |",
            "|---|---|---|---|",
        ]
    )
    opsd_store = []
    inter_store = []
    for stream in ("instrdialog", "instrdialogpp"):
        per_letter = {}
        for letter in ("R", "U", "P"):
            accs = _opsd_triplet(table, stream, letter, "seen_avg_score")
            fs = _opsd_triplet(table, stream, letter, "forgetting")
            if not accs or not fs:
                lines.append(f"| {DATA_LABEL[stream]} | {letter} | {DASH} | {DASH} |")
                continue
            lines.append(f"| {DATA_LABEL[stream]} | {letter} | {pm(accs)} | {pm(fs)} |")
            per_letter[letter] = (accs, fs)
            opsd_store.append((stream, letter, accs, fs))
        if "P" in per_letter and "R" in per_letter:
            ia = [per_letter["P"][0][i] - per_letter["R"][0][i] for i in range(3)]
            iff = [per_letter["P"][1][i] - per_letter["R"][1][i] for i in range(3)]
            inter_store.append((stream, ia, iff))
    lines.extend(
        [
            "",
            "| Data | I Acc | I F |",
            "|---|---|---|",
        ]
    )
    for stream, ia, iff in inter_store:
        lines.append(f"| {DATA_LABEL[stream]} | {pm(ia)} | {pm(iff)} |")
    return lines, stored, opsd_store, inter_store


def _seed_bits(values) -> str:
    return "、".join(f"seed {seed} {signed(value)}" for seed, value in zip(SEEDS, values))


def _effect_sentence(data: str, objective: str, label: str, accs, fs, who: str) -> str:
    acc_mean, acc_std = mean_std(accs)
    f_mean, f_std = mean_std(fs)
    acc_all_neg = all(v < 0 for v in accs)
    acc_all_pos = all(v > 0 for v in accs)
    f_all_down = all(v < 0 for v in fs)
    f_all_up = all(v > 0 for v in fs)
    crosses = any(v > 0 for v in accs) and any(v < 0 for v in accs)
    if acc_all_neg and f_all_up:
        prefix = "负向"
        tail = f"三颗 seed 都是 {who} 的准确率更低，而且遗忘都更高。遗忘没有下降。"
    elif acc_all_neg and f_all_down:
        prefix = "负向"
        tail = f"三颗 seed 的准确率都是 {who} 更低。遗忘三颗都下降，但准确率没有因此变好。"
    elif acc_all_neg:
        prefix = "负向"
        tail = f"三颗 seed 的准确率都是 {who} 更低。遗忘不是稳定下降。"
    elif acc_mean <= -0.01 or f_mean >= 0.01:
        prefix = "负向"
        tail = f"均值上 {who} 没有更好。P 不如对照的 seed 保持原符号。"
    elif acc_all_pos and f_all_down:
        prefix = "合理"
        tail = f"三颗 seed 都是 {who} 准确率更高且遗忘更低。"
    elif crosses or (abs(acc_mean) < 0.01 and abs(f_mean) < 0.015):
        prefix = "合理"
        tail = "三种子符号不完全一致，或均值接近 0。不能写成稳定优于对照。"
    else:
        prefix = "合理"
        tail = "方向要连同三种子一起看，不能只报均值的符号。"
    return (
        f"{prefix}:{data} {objective} 上 {label} 的 Acc 为 {acc_mean:+.3f}±{acc_std:.3f}"
        f"（{_seed_bits(accs)}），F 为 {f_mean:+.3f}±{f_std:.3f}（{_seed_bits(fs)}）。{tail}"
    )


def interpretation(table: dict, effects, opsd_store, inter_store) -> list:
    lines = [
        "## Q1–Q2",
        "",
        "Q1 问配额对齐之后，真余弦 PGDR（P）相对全历史随机（R）还有没有优势。对照量是 Δ_PGDR = P−R。",
        "Q2 问真余弦（P）相对 compat_uniform（U）还有没有优势。U 是格式兼容之后均匀抽 Related，再加上和 P 相同的 train-holdout 退化槽。U 不是 cosine prototype。对照量是 Δ_prototype = P−U。",
        "两问都只使用上面 36 个 v3c run。遗忘下降才算改善。P 不如 U、也不如 R 的格子按负数写下，不取绝对值，也不改成「持平」。",
        "",
        "### Q1 配额公平后 PGDR 相对随机",
        "",
    ]
    for stream, objective, _dpu_a, _dpu_f, dpr_a, dpr_f in effects:
        lines.append(_effect_sentence(DATA_LABEL[stream], OBJ_LABEL[objective], "Δ_PGDR", dpr_a, dpr_f, "P"))
        lines.append("")
    lines.extend(
        [
            "配额公平之后，P 没有稳定超过 R。InstrDialog 上三颗 seed 的符号相反，均值跨过 0。InstrDialog++ 的 CE 和 CE+OPSD 都是三颗 seed 的 P 准确率更低，而且遗忘都更高。这是负向结果，不是检索分数可以改写的结果。",
            "",
            "### Q2 真余弦相对 compat_uniform",
            "",
        ]
    )
    for stream, objective, dpu_a, dpu_f, _dpr_a, _dpr_f in effects:
        lines.append(_effect_sentence(DATA_LABEL[stream], OBJ_LABEL[objective], "Δ_prototype", dpu_a, dpu_f, "P"))
        lines.append("")
    lines.extend(
        [
            "真余弦没有稳定超过 compat_uniform。InstrDialog CE 只有 seed 456 是 P 更好，789 和 42 是 P 更差。InstrDialog CE+OPSD 只有 seed 42 的准确率是 P 更高，另外两颗是 P 更低，遗忘均值也没有下降。InstrDialog++ CE 三颗 seed 的准确率都是 P 低于 U，遗忘只是略降。InstrDialog++ CE+OPSD 的 seed 789 上 P 明显更差。这些差原样留在表里。",
            "",
            "### Δ_OPSD 与交互项 I",
            "",
            "Δ_OPSD = 同一选择器的 CE+OPSD 减去 CE。I = P 的 Δ_OPSD 减去 R 的 Δ_OPSD。I 的准确率为正，表示 OPSD 给 P 的准确率提升大于给 R 的提升。I 的遗忘为正，表示 OPSD 使 P 的遗忘比 R 增加得更多，这不是改善。",
            "",
        ]
    )
    for stream, letter, accs, fs in opsd_store:
        acc_mean, acc_std = mean_std(accs)
        f_mean, f_std = mean_std(fs)
        lines.append(
            f"{DATA_LABEL[stream]} {letter} 的 Δ_OPSD Acc {acc_mean:+.3f}±{acc_std:.3f}（{_seed_bits(accs)}），"
            f"F {f_mean:+.3f}±{f_std:.3f}（{_seed_bits(fs)}）。"
        )
    lines.append("")
    for stream, ia, iff in inter_store:
        acc_mean, acc_std = mean_std(ia)
        f_mean, f_std = mean_std(iff)
        lines.append(
            f"{DATA_LABEL[stream]} 的 I Acc {acc_mean:+.3f}±{acc_std:.3f}（{_seed_bits(ia)}），"
            f"I F {f_mean:+.3f}±{f_std:.3f}（{_seed_bits(iff)}）。"
        )
    lines.extend(
        [
            "",
            "InstrDialog 上 I 的准确率均值接近 0，seed 456 为负、seed 789 为正。InstrDialog++ 上 I 的遗忘均值为正，OPSD 没有把 P 的遗忘压到比 R 更低的增量上。不能把 I 写成 PGDR 与 OPSD 的正交互。",
            "",
        ]
    )
    return lines


def replay_audit() -> list:
    targets = {}
    cosine = {"R": [], "U": [], "P": []}
    missing = []
    for letter in ("R", "U", "P"):
        seen = set()
        for stream in ("instrdialog", "instrdialogpp"):
            for objective in ("ce", "opsd"):
                for seed in SEEDS:
                    name = cell_name(letter, objective, stream, seed)
                    path = RUNS / name / "replay_log.jsonl"
                    if not path.is_file():
                        missing.append(name)
                        continue
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        if int(row.get("segment_id", 0)) > 0 and row.get("n_replay_target") is not None:
                            seen.add(int(row["n_replay_target"]))
                        for value in row.get("selected_cosine_scores") or []:
                            if isinstance(value, (int, float)):
                                cosine[letter].append(float(value))
        targets[letter] = sorted(seen)
    lines = [
        "## 回放日志里的配额和余弦",
        "",
        "下面只统计 v3c `replay_log.jsonl`。segment 0 没有历史记忆，不拿来核对目标条数。",
        "",
    ]
    if missing:
        lines.append("缺少日志：" + ", ".join(missing) + "。缺日志的选择器不报告余弦均值。")
        lines.append("")
    for letter in ("R", "U", "P"):
        scores = cosine[letter]
        if scores:
            mean, std = mean_std(scores)
            lines.append(
                f"{letter} 在 segment_id>0 的 n_replay_target 取值 {targets[letter]}；"
                f"selected_cosine_scores 共 {len(scores)} 个，均值 {mean:.3f}，样本标准差 {std:.3f}。"
            )
        else:
            lines.append(f"{letter} 没有读到 selected_cosine_scores，不编分布数字。")
    lines.extend(
        [
            "",
            "P 的选中余弦均值高于 U 和 R，只说明被抽中的段在冻结指令向量上更接近当前段。它不是 Seen-Avg Acc，也不是遗忘下降。R 和 U 的余弦是日志里顺手记下的，选择时不用这个分数。",
            "",
        ]
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
    mean_body, _summaries = mean_lines(table)
    effect_body, effects, opsd_store, inter_store = effect_mean_lines(table)
    lines = [
        "# v3c 修正配额与真正原型检索",
        "",
        "本文件只含 `v3c_matched_budget` 的 36 个 run。一行是一个方法、一个数据集、一颗 seed。",
        "R 是全历史均匀，U 是格式兼容后的均匀 Related 加 train-holdout 退化，P 是显式 L2 余弦 Related 加同一套退化槽。U 不是 cosine prototype。",
        "mixed-batch ρ=0.20 在 45 条 current 上的目标是 11 条回放，实现比例 11/56，不是严格 20%。三种选择器共用这个整数。",
        "旧 `v3h_*` 的 9 对 8 不进入本文件的表、配对差或 Mean±Std。Protocol A 同样分开。",
        f"完成 {done}/36。队列日志里成功 run 的 elapsed 之和是 {gpu_seconds():.0f} 秒；这个数不是模型指标。",
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
    lines.extend(mean_body)
    lines.append("")
    lines.extend(effect_body)
    lines.append("")
    lines.extend(interpretation(table, effects, opsd_store, inter_store))
    lines.extend(replay_audit())
    if missing:
        lines.append("## 未完成")
        lines.append("")
        for name in missing:
            lines.append(f"- {name}: 无 final_metrics.json。")
        lines.append("")
    lines.extend(
        [
            "## 图",
            "",
            "图由 `results/figures/plot_v3c_corrected.py` 只根据已有 `final_metrics.json` 或 `replay_log.jsonl` 绘制。没有实测点的图不写文件。",
            "",
            "- `results/figures/prototype_vs_uniform.pdf`：同 seed 的 P−U。",
            "- `results/figures/budget_matched_comparison.pdf`：R/U/P 在两种目标上的实测均值，柱上标 n。",
            "- `results/figures/factorial_accuracy.pdf`：每颗 seed 的 CE 到 CE+OPSD 准确率。",
            "- `results/figures/factorial_forgetting.pdf`：同一因子的遗忘。",
            "- `results/figures/selected_cosine_distribution.pdf`：日志里的 selected L2 cosine。没有该字段就不画。",
            "- `results/figures/per_seed_effects.pdf`：每颗 seed 的 P−R 与 P−U。",
            "",
        ]
    )
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_plotter():
    spec = importlib.util.spec_from_file_location("plot_v3c_corrected", FIG_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    table = {}
    for name in all_runs():
        metrics = load_metrics(name)
        if metrics is not None:
            table[name] = metrics
    write_doc(table)
    figures = _load_plotter().write_figures(table)
    print(f"DOC {DOC}")
    print(f"COMPLETED {len(table)}/{len(all_runs())}")
    print(f"FIGURES {len(figures)}")
    for path in figures:
        print(path)
    if len(table) != 36 or len(figures) != 6:
        raise SystemExit(f"expected 36 runs and 6 figures, got {len(table)} and {len(figures)}")


if __name__ == "__main__":
    main()
