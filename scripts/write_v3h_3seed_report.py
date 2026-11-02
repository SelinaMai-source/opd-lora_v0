#!/usr/bin/env python3
"""Write the 24-row Protocol B table, 3-seed summary, and Q1–Q6. Missing = —."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

ROOT = Path("/root/research-repos/opd-lora_v3")
RUNS = Path("/root/autodl-tmp/rp_lora_v3_runs/runs")
OUT = ROOT / "docs" / "ACL_v3_Factorial_3Seed_Full_Results.md"
DASH = "—"
SEEDS = (456, 789, 42)
STREAMS = ("instrdialog", "instrdialogpp")
STREAM_LABEL = {"instrdialog": "InstrDialog", "instrdialogpp": "InstrDialog++"}
CELLS = ("rand_ce", "proto_ce", "rand_opsd", "proto_opsd")
KEYS = [
    ("seen_avg_score", "Seen-Avg Acc↑"),
    ("seen_avg_task_aware_score", "Seen-Avg Task-aware Acc↑"),
    ("forgetting", "Forgetting↓"),
    ("task_aware_forgetting", "Task-Aware Forgetting↓"),
    ("token_f1_mean", "Token F1↑"),
    ("rouge_l_mean", "ROUGE-L↑"),
    ("bleu_mean", "BLEU↑"),
    ("lcs_overlap_mean", "LCS Overlap↑"),
    ("current_score", "Current-Seg Acc↑"),
    ("current_task_aware_score", "Current-Seg Task-aware Acc↑"),
    ("task_aware_score_mean", "Task-aware Score Mean↑"),
]
HEADER = (
    "| 版本 | 内容 | Data | Seen-Avg Acc↑ | Seen-Avg Task-aware Acc↑ | Forgetting↓ | "
    "Task-Aware Forgetting↓ | Token F1↑ | ROUGE-L↑ | BLEU↑ | LCS Overlap↑ | "
    "Current-Seg Acc↑ | Current-Seg Task-aware Acc↑ | Task-aware Score Mean↑ | 结果解释 |"
)
SEP = "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def run_id(method: str, stream: str, seed: int) -> str:
    return f"v3h_{method}_{stream}_seed{seed}"


def load_eleven(method: str, stream: str, seed: int) -> dict[str, Optional[float]]:
    path = RUNS / run_id(method, stream, seed) / "final_metrics.json"
    out = {key: None for key, _ in KEYS}
    if not path.is_file():
        return out
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload.get("final") or {}
    for key, _ in KEYS:
        raw = final.get(f"eval.{key}")
        out[key] = float(raw) if isinstance(raw, (int, float)) else None
    return out


def fmt(value: Optional[float]) -> str:
    if value is None:
        return DASH
    return f"{value:.3f}"


def signed(value: Optional[float]) -> str:
    if value is None:
        return DASH
    return f"{value:+.3f}"


def content(method: str, seed: int) -> str:
    if method == "rand_ce":
        return (
            f"随机replay+CE,K=10/段,20% replay,λ=0,"
            f"train_holdout holdout_seed=456, seed {seed}"
        )
    if method == "proto_ce":
        return (
            f"prototype replay+CE,K=10/段,20% proto replay(10%近邻/10%退化),λ=0,"
            f"train_holdout holdout_seed=456, seed {seed}"
        )
    if method == "rand_opsd":
        return (
            f"随机replay+CE+λ=0.2 OPSD,K=10/段,20% replay,"
            f"train_holdout holdout_seed=456, seed {seed}"
        )
    if method == "proto_opsd":
        return (
            f"prototype replay+CE+λ=0.2 OPSD,K=10/段,20% proto replay(10%近邻/10%退化),"
            f"train_holdout holdout_seed=456, seed {seed}"
        )
    raise ValueError(method)


def pair_delta(method: str, stream: str, seed: int) -> str:
    me = load_eleven(method, stream, seed)
    acc = me["seen_avg_score"]
    forg = me["forgetting"]
    bits = [f"{run_id(method, stream, seed)};Acc {fmt(acc)},F {fmt(forg)}"]
    if method == "rand_ce":
        proto = load_eleven("proto_ce", stream, seed)
        opsd = load_eleven("rand_opsd", stream, seed)
        bits.append(
            f"同seed配对 proto_ce-本格 Acc {signed(_sub(proto['seen_avg_score'], acc))},"
            f"F {signed(_sub(proto['forgetting'], forg))}"
        )
        bits.append(
            f"同seed配对 rand_opsd-本格 Acc {signed(_sub(opsd['seen_avg_score'], acc))},"
            f"F {signed(_sub(opsd['forgetting'], forg))}"
        )
    elif method == "proto_ce":
        rand = load_eleven("rand_ce", stream, seed)
        opsd = load_eleven("proto_opsd", stream, seed)
        bits.append(
            f"同seed配对 vs rand_ce Acc {signed(_sub(acc, rand['seen_avg_score']))},"
            f"F {signed(_sub(forg, rand['forgetting']))}"
        )
        bits.append(
            f"同seed配对 proto_opsd-本格 Acc {signed(_sub(opsd['seen_avg_score'], acc))},"
            f"F {signed(_sub(opsd['forgetting'], forg))}"
        )
    elif method == "rand_opsd":
        ce = load_eleven("rand_ce", stream, seed)
        proto = load_eleven("proto_opsd", stream, seed)
        bits.append(
            f"同seed配对 vs rand_ce Acc {signed(_sub(acc, ce['seen_avg_score']))},"
            f"F {signed(_sub(forg, ce['forgetting']))}"
        )
        bits.append(
            f"同seed配对 proto_opsd-本格 Acc {signed(_sub(proto['seen_avg_score'], acc))},"
            f"F {signed(_sub(proto['forgetting'], forg))}"
        )
    else:
        ce = load_eleven("proto_ce", stream, seed)
        rand = load_eleven("rand_opsd", stream, seed)
        bits.append(
            f"同seed配对 vs proto_ce Acc {signed(_sub(acc, ce['seen_avg_score']))},"
            f"F {signed(_sub(forg, ce['forgetting']))}"
        )
        bits.append(
            f"同seed配对 vs rand_opsd Acc {signed(_sub(acc, rand['seen_avg_score']))},"
            f"F {signed(_sub(forg, rand['forgetting']))}"
        )
    bits.append("来源 final.eval.*")
    return ";".join(bits)


def _sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def explain(method: str, stream: str, seed: int) -> str:
    me = load_eleven(method, stream, seed)
    acc = me["seen_avg_score"]
    forg = me["forgetting"]
    if acc is None or forg is None:
        return f"异常:{run_id(method, stream, seed)};无 final_metrics.json,指标写—,来源缺失"
    prefix = "合理"
    if acc < 0.15:
        prefix = "异常"
    else:
        if method == "proto_ce":
            base = load_eleven("rand_ce", stream, seed)["seen_avg_score"]
            if base is not None and acc + 1e-9 < base - 0.02:
                prefix = "负向"
        elif method == "rand_opsd":
            base = load_eleven("rand_ce", stream, seed)["seen_avg_score"]
            if base is not None and acc + 1e-9 < base - 0.02:
                prefix = "负向"
        elif method == "proto_opsd":
            base = load_eleven("rand_opsd", stream, seed)["seen_avg_score"]
            proto_ce = load_eleven("proto_ce", stream, seed)["seen_avg_score"]
            worse_rand = base is not None and acc + 1e-9 < base - 0.02
            worse_ce = proto_ce is not None and acc + 1e-9 < proto_ce - 0.02
            if worse_rand and worse_ce:
                prefix = "负向"
    return f"{prefix}:{pair_delta(method, stream, seed)}"


def mean_std(values: list[Optional[float]]) -> str:
    xs = [v for v in values if v is not None]
    if not xs:
        return DASH
    if len(xs) == 1:
        return f"{xs[0]:.3f}±—"
    mu = sum(xs) / len(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return f"{mu:.3f}±{math.sqrt(var):.3f} (n={len(xs)})"


def deltas(stream: str, seed: int) -> dict[str, Optional[float]]:
    cells = {m: load_eleven(m, stream, seed) for m in CELLS}
    acc = {m: cells[m]["seen_avg_score"] for m in CELLS}
    forg = {m: cells[m]["forgetting"] for m in CELLS}

    def d(a, b):
        return None if a is None or b is None else a - b

    return {
        "d_pgdr_ce_acc": d(acc["proto_ce"], acc["rand_ce"]),
        "d_pgdr_opsd_acc": d(acc["proto_opsd"], acc["rand_opsd"]),
        "d_opsd_rand_acc": d(acc["rand_opsd"], acc["rand_ce"]),
        "d_opsd_proto_acc": d(acc["proto_opsd"], acc["proto_ce"]),
        "i_acc": None
        if any(acc[m] is None for m in CELLS)
        else acc["proto_opsd"] - acc["rand_opsd"] - acc["proto_ce"] + acc["rand_ce"],
        "d_pgdr_ce_f": d(forg["proto_ce"], forg["rand_ce"]),
        "d_pgdr_opsd_f": d(forg["proto_opsd"], forg["rand_opsd"]),
        "d_opsd_rand_f": d(forg["rand_opsd"], forg["rand_ce"]),
        "d_opsd_proto_f": d(forg["proto_opsd"], forg["proto_ce"]),
        "i_f": None
        if any(forg[m] is None for m in CELLS)
        else forg["proto_opsd"] - forg["rand_opsd"] - forg["proto_ce"] + forg["rand_ce"],
    }


def direction(values: list[Optional[float]], *, lower_better: bool = False) -> str:
    xs = [v for v in values if v is not None]
    if len(xs) < 3:
        return f"种子未齐(n={len(xs)})，不写一致性"
    pos = sum(v > 1e-9 for v in xs)
    neg = sum(v < -1e-9 for v in xs)
    if lower_better:
        good, bad = "下降", "上升"
        if neg == 3:
            return f"三种子均{good}"
        if pos == 3:
            return f"三种子均{bad}"
        return f"方向不一致({neg}降/{pos}升)"
    if pos == 3:
        return "三种子均为正"
    if neg == 3:
        return "三种子均为负"
    return f"方向不一致({pos}正/{neg}负)"


def table_rows() -> list[str]:
    rows = []
    for stream in STREAMS:
        for method in CELLS:
            for seed in SEEDS:
                metrics = load_eleven(method, stream, seed)
                cells = [fmt(metrics[key]) for key, _ in KEYS]
                rows.append(
                    "| "
                    + " | ".join(
                        [
                            "v3_factorial_heldout_3seed",
                            content(method, seed),
                            STREAM_LABEL[stream],
                            *cells,
                            explain(method, stream, seed),
                        ]
                    )
                    + " |"
                )
    return rows


def findings() -> str:
    chunks = []
    for q, title, body in (
        (
            "Q1",
            "PGDR 相对随机回放是否同向？",
            _q_pgdr(),
        ),
        (
            "Q2",
            "OPSD 相对 CE-only 是否同向？",
            _q_opsd(),
        ),
        (
            "Q3",
            "交互项 I 是否稳定？",
            _q_inter(),
        ),
        (
            "Q4",
            "遗忘是否随 PGDR / OPSD 下降？",
            _q_forget(),
        ),
        (
            "Q5",
            "两套 benchmark 故事是否一致？",
            _q_stream(),
        ),
        (
            "Q6",
            "Protocol B 相对 Protocol A 只能描述、不能归因泄漏？",
            _q_protocol(),
        ),
    ):
        chunks.append(f"### {q} {title}\n\n{body}")
    return "\n\n".join(chunks)


def _q_pgdr() -> str:
    lines = []
    for stream in STREAMS:
        ce = [deltas(stream, seed)["d_pgdr_ce_acc"] for seed in SEEDS]
        opsd = [deltas(stream, seed)["d_pgdr_opsd_acc"] for seed in SEEDS]
        lines.append(
            f"- {STREAM_LABEL[stream]} Δ_PGDR|CE Acc：{', '.join(signed(v) for v in ce)}；{direction(ce)}。"
            f"Δ_PGDR|OPSD Acc：{', '.join(signed(v) for v in opsd)}；{direction(opsd)}。"
        )
    lines.append("三种子不写显著性。不拿 seed 789/42 对 seed 456 说方法变好。")
    return "\n".join(lines)


def _q_opsd() -> str:
    lines = []
    for stream in STREAMS:
        rand = [deltas(stream, seed)["d_opsd_rand_acc"] for seed in SEEDS]
        proto = [deltas(stream, seed)["d_opsd_proto_acc"] for seed in SEEDS]
        lines.append(
            f"- {STREAM_LABEL[stream]} Δ_OPSD|random Acc：{', '.join(signed(v) for v in rand)}；{direction(rand)}。"
            f"Δ_OPSD|PGDR Acc：{', '.join(signed(v) for v in proto)}；{direction(proto)}。"
        )
    lines.append("三种子不写显著性。")
    return "\n".join(lines)


def _q_inter() -> str:
    lines = []
    for stream in STREAMS:
        acc = [deltas(stream, seed)["i_acc"] for seed in SEEDS]
        lines.append(
            f"- {STREAM_LABEL[stream]} I_Acc：{', '.join(signed(v) for v in acc)}；{direction(acc)}。"
        )
    lines.append("I = proto_opsd − rand_opsd − proto_ce + rand_ce。缺一格则该 seed 为 —。")
    return "\n".join(lines)


def _q_forget() -> str:
    lines = ["遗忘有利方向是下降。"]
    for stream in STREAMS:
        pgdr_ce = [deltas(stream, seed)["d_pgdr_ce_f"] for seed in SEEDS]
        opsd_rand = [deltas(stream, seed)["d_opsd_rand_f"] for seed in SEEDS]
        lines.append(
            f"- {STREAM_LABEL[stream]} Δ_PGDR|CE Forgetting：{', '.join(signed(v) for v in pgdr_ce)}；"
            f"{direction(pgdr_ce, lower_better=True)}。"
            f"Δ_OPSD|random Forgetting：{', '.join(signed(v) for v in opsd_rand)}；"
            f"{direction(opsd_rand, lower_better=True)}。"
        )
    return "\n".join(lines)


def _q_stream() -> str:
    return (
        "不一致，只能分流域写。InstrDialog++ 上 Δ_PGDR|CE Acc 三种子为正、"
        "Δ_PGDR|CE 与 Δ_OPSD|random 遗忘三种子均下降；InstrDialog 上 Acc 的 PGDR/OPSD "
        "主效应都有 1 个 seed 反号（seed 42），只有 Δ_OPSD|random 遗忘三种子同向下降。"
        "不能把一套流的符号套到另一套，也不能写成总效应。"
    )


def _q_protocol() -> str:
    return (
        "Protocol A（`eval_split_monitor`）只作描述性对照。Protocol B 每段 fit 从 50 降到 45，"
        "监控 5 条不进梯度；uniform 回放 9、proto 回放 8。A/B 差值混杂了消泄漏与训练条数变化，"
        "不能把差值全部写成泄漏效应。近邻槽仍是格式兼容后的均匀抽样，不是余弦检索。"
    )


def summary_tables() -> str:
    blocks = []
    for stream in STREAMS:
        rows = [
            "| Cell | Seen-Avg Acc Mean±SD | Forgetting Mean±SD | n Acc |",
            "|---|---|---|---|",
        ]
        for method in CELLS:
            accs = [load_eleven(method, stream, seed)["seen_avg_score"] for seed in SEEDS]
            fors = [load_eleven(method, stream, seed)["forgetting"] for seed in SEEDS]
            n = sum(v is not None for v in accs)
            rows.append(f"| {method} | {mean_std(accs)} | {mean_std(fors)} | {n} |")
        rows.append("")
        rows.append("| Seed | Δ_PGDR\\|CE Acc | Δ_PGDR\\|OPSD Acc | Δ_OPSD\\|rand Acc | Δ_OPSD\\|PGDR Acc | I Acc | Δ_PGDR\\|CE F | Δ_OPSD\\|rand F | I F |")
        rows.append("|---|---|---|---|---|---|---|---|---|")
        for seed in SEEDS:
            d = deltas(stream, seed)
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(seed),
                        signed(d["d_pgdr_ce_acc"]),
                        signed(d["d_pgdr_opsd_acc"]),
                        signed(d["d_opsd_rand_acc"]),
                        signed(d["d_opsd_proto_acc"]),
                        signed(d["i_acc"]),
                        signed(d["d_pgdr_ce_f"]),
                        signed(d["d_opsd_rand_f"]),
                        signed(d["i_f"]),
                    ]
                )
                + " |"
            )
        blocks.append(f"### {STREAM_LABEL[stream]}\n\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def main() -> None:
    measured = 0
    for stream in STREAMS:
        for method in CELLS:
            for seed in SEEDS:
                if load_eleven(method, stream, seed)["seen_avg_score"] is not None:
                    measured += 1
    resume = (
        "```bash\n"
        "cd /root/research-repos/opd-lora_v3\n"
        "export PYTHONPATH=/root/research-repos/opd-lora_v3\n"
        "export HF_HOME=/root/autodl-tmp/hf_cache\n"
        "export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache\n"
        "export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub\n"
        "export TORCH_HOME=/root/autodl-tmp/torch_cache\n"
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
        "/root/autodl-tmp/conda_envs/rp_lora_v0/bin/python scripts/run_v3h_p0_3seed_queue.py\n"
        "```"
    )
    text = f"""# ACL v3 Factorial 3-Seed Full Results

Protocol B only（`monitor: train_holdout`，`holdout_seed: 456`，`eval_normalization.format_normalize: true`，`benchmark/*_multiref.json`）。版本列统一 `v3_factorial_heldout_3seed`。24 行 = 4 方法 × 2 数据 × 3 seed。缺 `final_metrics.json` 的格写 {DASH}，不估数。

复用 8 个已验证 seed 456；本文件不把 Protocol A 行写进主表。科学限制保持：每段 fit 45 / 监控 5；uniform 回放 9、proto 8；近邻槽不是余弦检索。

已测 {measured} / 24。

## Protocol B 24 行

{HEADER}
{SEP}
{chr(10).join(table_rows())}

未完成行若存在，续跑整队：

{resume}

单条命令见 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。不要覆盖已有目录。

## 3-Seed Statistical Summary

样本标准差用 n−1。误差棒和图只在三种子都齐时画。三种子不写显著性，只看方向是否一致。遗忘有利方向是下降。Δ 与 I 都在同 seed、同数据内计算，不拿 789/42 对 456 说方法变好。

{summary_tables()}

## Research Findings and Interpretation

{findings()}
"""
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} measured={measured}/24")


if __name__ == "__main__":
    main()
