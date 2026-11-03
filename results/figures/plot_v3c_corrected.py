#!/usr/bin/env python3
"""v3c figures from measured final_metrics.json and replay_log.jsonl.

A missing run is omitted. No line is drawn through an unmeasured point.
A panel with no measured points is not written. Protocol A and v3h are not read.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGS = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("OPD_RUNS_DIR", "/root/autodl-tmp/rp_lora_v3_runs/runs"))
STREAMS = ("instrdialog", "instrdialogpp")
STREAM_LABEL = {"instrdialog": "InstrDialog", "instrdialogpp": "InstrDialog++"}
LETTERS = ("R", "U", "P")
SEEDS = (456, 789, 42)
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


def _name(letter: str, objective: str, stream: str, seed: int) -> str:
    return f"v3c_{letter}_{objective}_{stream}_seed{seed}"


def load_records(runs: Path = RUNS) -> Dict[str, Dict[str, float]]:
    records: Dict[str, Dict[str, float]] = {}
    for stream in STREAMS:
        for letter in LETTERS:
            for objective in ("ce", "opsd"):
                for seed in SEEDS:
                    name = _name(letter, objective, stream, seed)
                    path = runs / name / "final_metrics.json"
                    if not path.is_file():
                        continue
                    final = json.loads(path.read_text(encoding="utf-8")).get("final") or {}
                    row = {}
                    ok = True
                    for key in KEYS:
                        raw = final.get(f"eval.{key}")
                        if not isinstance(raw, (int, float)):
                            ok = False
                            break
                        row[key] = float(raw)
                    if ok:
                        records[name] = row
    return records


def write_figures(records: Dict[str, Dict[str, float]]) -> List[str]:
    FIGS.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    if not any(name.startswith("v3c_") for name in records):
        return written
    for maker in (
        lambda: _budget_bars(records),
        lambda: _proto_vs_uniform(records),
        lambda: _factorial(records, "seen_avg_score", "factorial_accuracy.pdf", "Seen-Avg Acc"),
        lambda: _factorial(records, "forgetting", "factorial_forgetting.pdf", "Forgetting (lower is better)"),
        _cosine_plot,
        lambda: _per_seed(records),
    ):
        path = maker()
        if path:
            written.append(path)
    return written


def _save(fig, filename: str) -> str:
    fig.tight_layout()
    path = FIGS / filename
    fig.savefig(path)
    plt.close(fig)
    return str(path.relative_to(ROOT))


def _budget_bars(records: Dict[str, Dict[str, float]]) -> Optional[str]:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=False)
    panels = (
        (0, 0, "instrdialog", "seen_avg_score", "Acc"),
        (0, 1, "instrdialog", "forgetting", "Forgetting"),
        (1, 0, "instrdialogpp", "seen_avg_score", "Acc"),
        (1, 1, "instrdialogpp", "forgetting", "Forgetting"),
    )
    colors = {"R": "#4C78A8", "U": "#F58518", "P": "#54A24B"}
    any_bar = False
    for row, col, stream, key, title in panels:
        ax = axes[row][col]
        xs, heights, cols, labels = [], [], [], []
        i = 0
        for objective in ("ce", "opsd"):
            for letter in LETTERS:
                vals = [
                    records[_name(letter, objective, stream, seed)][key]
                    for seed in SEEDS
                    if _name(letter, objective, stream, seed) in records
                ]
                if not vals:
                    continue
                any_bar = True
                xs.append(i)
                heights.append(sum(vals) / len(vals))
                cols.append(colors[letter])
                labels.append(f"{letter}-{objective}\nn={len(vals)}")
                i += 1
            i += 0.6
        if heights:
            ax.bar(xs, heights, color=cols)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{STREAM_LABEL[stream]} {title}")
    if not any_bar:
        plt.close(fig)
        return None
    fig.suptitle("Matched budget (measured seeds only)")
    return _save(fig, "budget_matched_comparison.pdf")


def _proto_vs_uniform(records: Dict[str, Dict[str, float]]) -> Optional[str]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    drew = False
    for ax, key, title in (
        (axes[0], "seen_avg_score", "Acc, P minus U"),
        (axes[1], "forgetting", "Forgetting, P minus U"),
    ):
        xs, ys, names = [], [], []
        i = 0
        for stream in STREAMS:
            for objective in ("ce", "opsd"):
                for seed in SEEDS:
                    p = records.get(_name("P", objective, stream, seed))
                    u = records.get(_name("U", objective, stream, seed))
                    if p is None or u is None:
                        continue
                    drew = True
                    xs.append(i)
                    ys.append(p[key] - u[key])
                    names.append(f"{STREAM_LABEL[stream][:2]}-{objective}-{seed}")
                    i += 1
        if ys:
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.bar(xs, ys, color="#54A24B")
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_title(title)
    if not drew:
        plt.close(fig)
        return None
    return _save(fig, "prototype_vs_uniform.pdf")


def _factorial(records: Dict[str, Dict[str, float]], key: str, filename: str, title: str) -> Optional[str]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    markers = {"R": "o", "U": "s", "P": "^"}
    drew = False
    for ax, stream in zip(axes, STREAMS):
        for letter in LETTERS:
            for seed in SEEDS:
                xs, ys = [], []
                for x, objective in ((0, "ce"), (1, "opsd")):
                    row = records.get(_name(letter, objective, stream, seed))
                    if row is None:
                        continue
                    xs.append(x + (SEEDS.index(seed) - 1) * 0.05)
                    ys.append(row[key])
                if not ys:
                    continue
                drew = True
                ax.plot(xs, ys, marker=markers[letter], linestyle="-", label=f"{letter}-{seed}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CE", "CE+OPSD"])
        ax.set_title(STREAM_LABEL[stream])
        ax.legend(fontsize=6, ncol=2)
    if not drew:
        plt.close(fig)
        return None
    fig.suptitle(title)
    return _save(fig, filename)


def _cosine_plot() -> Optional[str]:
    series = {"R": [], "U": [], "P": []}
    found = False
    for stream in STREAMS:
        for objective in ("ce", "opsd"):
            for seed in SEEDS:
                for letter in LETTERS:
                    path = RUNS / _name(letter, objective, stream, seed) / "replay_log.jsonl"
                    if not path.is_file():
                        continue
                    for line in path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        scores = row.get("selected_cosine_scores") or []
                        if scores:
                            found = True
                        series[letter].extend(float(x) for x in scores if isinstance(x, (int, float)))
    if not found:
        return None
    data, labels = [], []
    for letter in LETTERS:
        if series[letter]:
            data.append(series[letter])
            labels.append(f"{letter} n={len(series[letter])}")
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("Selected L2 cosine")
    ax.set_title("Selected replay cosine (measured logs only)")
    return _save(fig, "selected_cosine_distribution.pdf")


def _per_seed(records: Dict[str, Dict[str, float]]) -> Optional[str]:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    drew = False
    for ax, key, title in (
        (axes[0], "seen_avg_score", "Acc effect"),
        (axes[1], "forgetting", "Forgetting effect"),
    ):
        i = 0
        labels = []
        for stream in STREAMS:
            for objective in ("ce", "opsd"):
                for seed in SEEDS:
                    p = records.get(_name("P", objective, stream, seed))
                    u = records.get(_name("U", objective, stream, seed))
                    r = records.get(_name("R", objective, stream, seed))
                    if p is None or r is None:
                        continue
                    drew = True
                    ax.scatter(i, p[key] - r[key], marker="^", color="#54A24B", label="P-R" if i == 0 else None)
                    if u is not None:
                        ax.scatter(i, p[key] - u[key], marker="o", color="#F58518", label="P-U" if i == 0 else None)
                    labels.append(f"{stream[:2]}-{objective}-{seed}")
                    i += 1
        if labels:
            ax.axhline(0.0, color="black", linewidth=0.6)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_title(title)
        ax.legend(fontsize=7)
    if not drew:
        plt.close(fig)
        return None
    return _save(fig, "per_seed_effects.pdf")


def main() -> None:
    sys.path.insert(0, str(ROOT))
    records = load_records()
    written = write_figures(records)
    print(f"RECORDS {len(records)}")
    print(f"FIGURES {len(written)}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
