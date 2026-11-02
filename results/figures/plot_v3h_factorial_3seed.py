#!/usr/bin/env python3
"""Six Protocol B 3-seed figures. Missing cells stay blank; no imputed numbers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/root/research-repos/opd-lora_v3")
RUNS = Path("/root/autodl-tmp/rp_lora_v3_runs/runs")
FIGS = ROOT / "results" / "figures"
SEEDS = (456, 789, 42)
STREAMS = ("instrdialog", "instrdialogpp")
CELLS = ("rand_ce", "proto_ce", "rand_opsd", "proto_opsd")
CELL_LABEL = {
    "rand_ce": "rand+CE",
    "proto_ce": "PGDR+CE",
    "rand_opsd": "rand+OPSD",
    "proto_opsd": "PGDR+OPSD",
}
STREAM_LABEL = {"instrdialog": "InstrDialog", "instrdialogpp": "InstrDialog++"}
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


def run_name(method: str, stream: str, seed: int) -> str:
    return f"v3h_{method}_{stream}_seed{seed}"


def load_key(method: str, stream: str, seed: int, key: str) -> Optional[float]:
    path = RUNS / run_name(method, stream, seed) / "final_metrics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = (payload.get("final") or {}).get(f"eval.{key}")
    return float(raw) if isinstance(raw, (int, float)) else None


def sample_mean_std(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float], int]:
    xs = [v for v in values if v is not None]
    if not xs:
        return None, None, 0
    arr = np.array(xs, dtype=float)
    if len(arr) == 1:
        return float(arr[0]), None, 1
    return float(arr.mean()), float(arr.std(ddof=1)), len(arr)


def delta_pgdr(stream: str, seed: int, key: str, *, opsd: bool) -> Optional[float]:
    left = "proto_opsd" if opsd else "proto_ce"
    right = "rand_opsd" if opsd else "rand_ce"
    a = load_key(left, stream, seed, key)
    b = load_key(right, stream, seed, key)
    if a is None or b is None:
        return None
    return a - b


def delta_opsd(stream: str, seed: int, key: str, *, proto: bool) -> Optional[float]:
    left = "proto_opsd" if proto else "rand_opsd"
    right = "proto_ce" if proto else "rand_ce"
    a = load_key(left, stream, seed, key)
    b = load_key(right, stream, seed, key)
    if a is None or b is None:
        return None
    return a - b


def interaction(stream: str, seed: int, key: str) -> Optional[float]:
    cells = {m: load_key(m, stream, seed, key) for m in CELLS}
    if any(v is None for v in cells.values()):
        return None
    return cells["proto_opsd"] - cells["rand_opsd"] - cells["proto_ce"] + cells["rand_ce"]


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=160)
    plt.close(fig)


def plot_mean_std(key: str, ylabel: str, outfile: str, *, negate: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    x = np.arange(len(CELLS))
    for ax, stream in zip(axes, STREAMS):
        means, stds, ns = [], [], []
        for method in CELLS:
            vals = [load_key(method, stream, seed, key) for seed in SEEDS]
            if negate:
                vals = [None if v is None else -v for v in vals]
            mean, std, n = sample_mean_std(vals)
            means.append(mean)
            stds.append(0.0 if std is None else std)
            ns.append(n)
        plot_means = [0.0 if m is None else m for m in means]
        yerr = [s if n == 3 else 0.0 for s, n in zip(stds, ns)]
        colors = ["#9e9e9e" if n == 0 else "#2b8cbe" if n == 3 else "#a6bddb" for n in ns]
        ax.bar(x, plot_means, yerr=yerr, capsize=4, color=colors, ecolor="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABEL[m] for m in CELLS], rotation=20, ha="right")
        ax.set_title(STREAM_LABEL[stream])
        ax.set_ylabel(ylabel)
        ax.axhline(0.0, color="#666666", linewidth=0.6)
    fig.suptitle("Protocol B · seeds 456/789/42 · error bar only when n=3")
    _save(fig, outfile)


def plot_effect_by_seed(kind: str, outfile: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    width = 0.35
    for ax, stream in zip(axes, STREAMS):
        xs = np.arange(len(SEEDS))
        if kind == "pgdr":
            a = [delta_pgdr(stream, seed, "seen_avg_score", opsd=False) for seed in SEEDS]
            b = [delta_pgdr(stream, seed, "seen_avg_score", opsd=True) for seed in SEEDS]
            labels = ("ΔAcc PGDR | CE", "ΔAcc PGDR | OPSD")
            title = "PGDR − random, same seed"
        else:
            a = [delta_opsd(stream, seed, "seen_avg_score", proto=False) for seed in SEEDS]
            b = [delta_opsd(stream, seed, "seen_avg_score", proto=True) for seed in SEEDS]
            labels = ("ΔAcc OPSD | random", "ΔAcc OPSD | PGDR")
            title = "OPSD − CE, same seed"
        def bars(vals, shift, color, label):
            plot = [0.0 if v is None else v for v in vals]
            colors = [color if v is not None else "#d9d9d9" for v in vals]
            ax.bar(xs + shift, plot, width, color=colors, label=label)

        bars(a, -width / 2, "#1b9e77", labels[0])
        bars(b, width / 2, "#d95f02", labels[1])
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in SEEDS])
        ax.set_title(f"{STREAM_LABEL[stream]} · {title}")
        ax.axhline(0.0, color="#666666", linewidth=0.6)
        ax.set_ylabel("Δ Seen-Avg Acc")
        ax.legend(fontsize=8)
    _save(fig, outfile)


def plot_interaction(outfile: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    xs = np.arange(len(SEEDS))
    for ax, stream in zip(axes, STREAMS):
        acc = [interaction(stream, seed, "seen_avg_score") for seed in SEEDS]
        forg = [interaction(stream, seed, "forgetting") for seed in SEEDS]
        ax.bar(xs - 0.18, [0.0 if v is None else v for v in acc], 0.36, color="#2b8cbe", label="I Acc")
        ax.bar(xs + 0.18, [0.0 if v is None else v for v in forg], 0.36, color="#f03b20", label="I Forgetting")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(s) for s in SEEDS])
        ax.set_title(f"{STREAM_LABEL[stream]} · I = proto_opsd−rand_opsd−proto_ce+rand_ce")
        ax.axhline(0.0, color="#666666", linewidth=0.6)
        ax.legend(fontsize=8)
    _save(fig, outfile)


def plot_eleven(outfile: str) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(12.5, 11.0))
    axes = axes.ravel()
    x = np.arange(len(CELLS))
    for i, (key, label) in enumerate(KEYS):
        ax = axes[i]
        width = 0.38
        for j, stream in enumerate(STREAMS):
            means, stds, ns = [], [], []
            for method in CELLS:
                vals = [load_key(method, stream, seed, key) for seed in SEEDS]
                mean, std, n = sample_mean_std(vals)
                means.append(0.0 if mean is None else mean)
                stds.append(0.0 if std is None or n < 3 else std)
                ns.append(n)
            ax.bar(
                x + (-0.19 if j == 0 else 0.19),
                means,
                width,
                yerr=stds,
                capsize=3,
                label=STREAM_LABEL[stream] if i == 0 else None,
                color="#2b8cbe" if j == 0 else "#fdae6b",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([CELL_LABEL[m] for m in CELLS], rotation=25, ha="right", fontsize=7)
        ax.set_title(label, fontsize=9)
    axes[-1].axis("off")
    fig.legend(loc="lower right")
    fig.suptitle("Protocol B 11 metrics · mean over available seeds · error bar only when n=3")
    _save(fig, outfile)


def main() -> None:
    plot_mean_std("seen_avg_score", "Seen-Avg Acc", "factorial_accuracy_mean_std.pdf")
    plot_mean_std(
        "forgetting",
        "−Forgetting (larger better)",
        "factorial_forgetting_mean_std.pdf",
        negate=True,
    )
    plot_effect_by_seed("pgdr", "pgdr_effect_by_seed.pdf")
    plot_effect_by_seed("opsd", "opsd_effect_by_seed.pdf")
    plot_interaction("factorial_interaction.pdf")
    plot_eleven("eleven_metrics_comparison.pdf")
    print(f"wrote figures under {FIGS}")


if __name__ == "__main__":
    main()
