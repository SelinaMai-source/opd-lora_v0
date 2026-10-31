#!/usr/bin/env python3
"""Build measured-only tables, interactions, and figures. Missing cells stay em dash."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/root/research-repos/opd-lora_v3")
NEW_RUNS = Path("/root/autodl-tmp/rp_lora_v3_runs/runs")
OLD_RUNS = Path("/root/autodl-tmp/rp_lora_v0/results/runs")
NEW_LOGS = Path("/root/autodl-tmp/rp_lora_v3_runs/logs")
DOCS = ROOT / "docs"
FIGS = DOCS / "figures"
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

CELLS = ("rand_ce", "proto_ce", "rand_opsd", "proto_opsd")
STREAMS = ("instrdialog", "instrdialogpp")


def fmt(value: Optional[float]) -> str:
    if value is None:
        return DASH
    return f"{value:.3f}"


def find_metrics(run_name: str) -> Optional[Path]:
    for base in (NEW_RUNS, OLD_RUNS):
        path = base / run_name / "final_metrics.json"
        if path.is_file():
            return path
    return None


def load_eleven(run_name: str) -> Dict[str, Optional[float]]:
    path = find_metrics(run_name)
    out: Dict[str, Optional[float]] = {key: None for key, _ in KEYS}
    out["_path"] = None  # type: ignore
    if path is None:
        return out
    payload = json.loads(path.read_text(encoding="utf-8"))
    final = payload.get("final") or {}
    for key, _ in KEYS:
        raw = final.get(f"eval.{key}")
        out[key] = float(raw) if isinstance(raw, (int, float)) else None
    out["_path"] = str(path)  # type: ignore
    return out


def run_name(prefix: str, method: str, stream: str, seed: int) -> str:
    return f"{prefix}_{method}_{stream}_seed{seed}"


def interaction(cells: Dict[str, Optional[float]]) -> Optional[float]:
    needed = ("rand_ce", "proto_ce", "rand_opsd", "proto_opsd")
    if any(cells.get(name) is None for name in needed):
        return None
    return float(cells["proto_opsd"] - cells["rand_opsd"] - cells["proto_ce"] + cells["rand_ce"])


def main_effect(cells: Dict[str, Optional[float]], kind: str) -> Optional[float]:
    if any(cells.get(name) is None for name in CELLS):
        return None
    if kind == "proto":
        return 0.5 * (
            (cells["proto_opsd"] - cells["rand_opsd"]) + (cells["proto_ce"] - cells["rand_ce"])
        )
    if kind == "opsd":
        return 0.5 * (
            (cells["proto_opsd"] - cells["proto_ce"]) + (cells["rand_opsd"] - cells["rand_ce"])
        )
    raise ValueError(kind)


def matrix_for(prefix: str, stream: str, seed: int, key: str) -> Dict[str, Optional[float]]:
    return {method: load_eleven(run_name(prefix, method, stream, seed))[key] for method in CELLS}


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def collect_grid() -> List[dict]:
    specs = []
    for stream in STREAMS:
        for method in CELLS:
            specs.append(("v3", method, stream, 456, "eval_split_monitor"))
            if method != "proto_opsd":
                for seed in (789, 42):
                    specs.append(("v3", method, stream, seed, "eval_split_monitor"))
            else:
                for seed in (789, 42):
                    specs.append(("v3", method, stream, seed, "eval_split_monitor"))
        for method in CELLS:
            specs.append(("v3h", method, stream, 456, "train_holdout"))
    # unique preserve order
    seen = set()
    rows = []
    for prefix, method, stream, seed, monitor in specs:
        name = run_name(prefix, method, stream, seed)
        if name in seen:
            continue
        seen.add(name)
        metrics = load_eleven(name)
        rows.append(
            {
                "run": name,
                "prefix": prefix,
                "method": method,
                "stream": stream,
                "seed": seed,
                "monitor": monitor,
                "path": metrics.get("_path"),
                **{key: metrics[key] for key, _ in KEYS},
            }
        )
    return rows


def write_csv(rows: List[dict]) -> None:
    path = DOCS / "v3_eleven_metrics.csv"
    fields = ["run", "prefix", "method", "stream", "seed", "monitor", "path"] + [key for key, _ in KEYS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fields}
            for key, _ in KEYS:
                if out[key] is None:
                    out[key] = ""
            writer.writerow(out)


def eleven_md(rows: List[dict], title_filter) -> str:
    headers = ["run", "monitor"] + [label for _, label in KEYS]
    body = []
    for row in rows:
        if not title_filter(row):
            continue
        body.append([row["run"], row["monitor"]] + [fmt(row[key]) for key, _ in KEYS])
    if not body:
        return "_无实测行_"
    return md_table(headers, body)


def interaction_rows(prefix: str, seed: int) -> List[List[str]]:
    body = []
    for stream in STREAMS:
        for key, label in KEYS:
            cells = matrix_for(prefix, stream, seed, key)
            raw = interaction(cells)
            flipped = None if raw is None or key not in {"forgetting", "task_aware_forgetting"} else -raw
            body.append(
                [
                    stream,
                    label,
                    fmt(cells["rand_ce"]),
                    fmt(cells["proto_ce"]),
                    fmt(cells["rand_opsd"]),
                    fmt(cells["proto_opsd"]),
                    fmt(raw),
                    fmt(main_effect(cells, "proto")),
                    fmt(main_effect(cells, "opsd")),
                    fmt(flipped) if key in {"forgetting", "task_aware_forgetting"} else "",
                ]
            )
    return body


def plot_bars(prefix: str, seed: int, key: str, ylabel: str, outfile: Path, *, negate: bool = False) -> None:
    labels = []
    values = []
    for stream in STREAMS:
        cells = matrix_for(prefix, stream, seed, key)
        for method in CELLS:
            labels.append(f"{stream[:2]}:{method}")
            value = cells[method]
            if value is None:
                values.append(None)
            else:
                values.append(-value if negate else value)
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = list(range(len(labels)))
    plot_vals = [0.0 if v is None else v for v in values]
    colors = ["#bdbdbd" if v is None else "#2c7fb8" for v in values]
    ax.bar(xs, plot_vals, color=colors)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    title = f"{prefix} seed {seed} {key}"
    if negate:
        title += " (sign-flipped, larger better)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outfile, dpi=140)
    plt.close(fig)


def trajectory(run_name_str: str) -> List[dict]:
    base = None
    for root in (NEW_RUNS, OLD_RUNS):
        if (root / run_name_str).is_dir():
            base = root / run_name_str
            break
    if base is None:
        return []
    series: Dict[int, List[float]] = {}
    for path in sorted(base.glob("segment_*/eval_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = ((payload.get("extra") or {}).get("per_segment_accuracy")) or []
        for row in rows:
            series.setdefault(int(row["segment_id"]), []).append(float(row["accuracy"]))
    notes = []
    for sid, accs in series.items():
        if len(accs) < 2:
            continue
        peak = max(accs)
        end = accs[-1]
        dropped = any(accs[i] + 1e-9 < accs[i - 1] for i in range(1, len(accs)))
        recovered = any(
            accs[i] > accs[j] + 1e-9 and any(accs[k] + 1e-9 < accs[j] for k in range(j))
            for j in range(len(accs))
            for i in range(j + 1, len(accs))
        )
        kind = "flat"
        if peak > 0 and end + 1e-9 < peak and dropped:
            kind = "learned_then_forgotten"
        if recovered and end + 1e-9 > min(accs):
            kind = "forgotten_then_recovered" if kind == "flat" else kind + "+recovered"
        if peak > 0 and abs(end - peak) < 1e-9 and not dropped:
            kind = "kept"
        notes.append({"segment_id": sid, "kind": kind, "accs": [round(a, 3) for a in accs]})
    return notes


def real_cases(run_name_str: str, limit: int = 2) -> List[dict]:
    base = None
    for root in (NEW_RUNS, OLD_RUNS):
        if (root / run_name_str / "eval_debug").is_dir():
            base = root / run_name_str / "eval_debug"
            break
    if base is None:
        return []
    files = sorted(base.glob("eval_segment_*.json"))
    if not files:
        return []
    payload = json.loads(files[-1].read_text(encoding="utf-8"))
    examples = payload.get("examples") or []
    picked = []
    for want_match in (True, False):
        for ex in examples:
            if bool(ex.get("match")) is want_match:
                picked.append(
                    {
                        "run": run_name_str,
                        "segment_id": ex.get("source_segment_id"),
                        "match": bool(ex.get("match")),
                        "instruction": str(ex.get("instruction", ""))[:180],
                        "input": str(ex.get("input_text", ""))[:180],
                        "gold": str(ex.get("gold_output", ""))[:180],
                        "pred": str(ex.get("raw_generated_output", ""))[:180],
                    }
                )
                break
        if len(picked) >= limit:
            break
    return picked


def train_scalars(run_name_str: str) -> Dict[str, Optional[float]]:
    path = None
    for root in (NEW_RUNS, OLD_RUNS):
        candidate = root / run_name_str / "metrics.jsonl"
        if candidate.is_file():
            path = candidate
            break
    empty = {"ce": None, "opsd": None, "rollout_tokens": None, "grad_norm": None, "n": 0}
    if path is None:
        return empty
    ce, opsd, rollout, grad = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for dest, key in (
            (ce, "train.train.ce"),
            (opsd, "train.train.opsd"),
            (rollout, "train.rollout_mean_tokens"),
            (grad, "train.grad_norm"),
        ):
            if isinstance(row.get(key), (int, float)):
                dest.append(float(row[key]))
    def avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    return {"ce": avg(ce), "opsd": avg(opsd), "rollout_tokens": avg(rollout), "grad_norm": avg(grad), "n": len(ce)}


def timing_rows() -> List[List[str]]:
    body = []
    if not NEW_LOGS.is_dir():
        return body
    for start in sorted(NEW_LOGS.glob("*.start")):
        name = start.name[: -len(".start")]
        end = NEW_LOGS / f"{name}.end"
        elapsed = NEW_LOGS / f"{name}.elapsed_sec"
        gpu = NEW_LOGS / f"{name}.gpu_after"
        sec = elapsed.read_text(encoding="utf-8").strip() if elapsed.is_file() else DASH
        body.append(
            [
                name,
                start.read_text(encoding="utf-8").strip(),
                end.read_text(encoding="utf-8").strip() if end.is_file() else DASH,
                sec,
                gpu.read_text(encoding="utf-8").strip() if gpu.is_file() else DASH,
            ]
        )
    return body


def baseline_inventory() -> List[str]:
    names = []
    root = ROOT / "baselines"
    for path in sorted(root.rglob("method.py")):
        if "official" in path.parts:
            continue
        names.append(str(path.relative_to(ROOT)))
    return names


def write_docs(rows: List[dict]) -> None:
    inter_headers = [
        "Data",
        "metric",
        "rand_ce",
        "proto_ce",
        "rand_opsd",
        "proto_opsd",
        "interaction",
        "proto main",
        "OPSD main",
        "forgetting sign-flip",
    ]
    old_inter = md_table(inter_headers, interaction_rows("v3", 456))
    new_inter = md_table(inter_headers, interaction_rows("v3h", 456))
    seed_rows = []
    for seed in (789, 42):
        seed_rows.append(f"### seed {seed} 旧协议（缺格为 —）\n\n" + md_table(inter_headers, interaction_rows("v3", seed)))
    case_lines = []
    for name in (
        "v3_proto_opsd_instrdialog_seed456",
        "v3h_proto_opsd_instrdialog_seed456",
    ):
        for case in real_cases(name):
            case_lines.append(
                f"- `{name}` seg {case['segment_id']} match={case['match']}\n"
                f"  - gold: {case['gold']}\n"
                f"  - pred: {case['pred']}"
            )
    if not case_lines:
        case_lines.append("- —")
    traj_lines = []
    for name in (
        "v3_proto_opsd_instrdialog_seed456",
        "v3_rand_opsd_instrdialog_seed456",
        "v3h_proto_opsd_instrdialog_seed456",
    ):
        notes = trajectory(name)
        counts: Dict[str, int] = {}
        for note in notes:
            counts[note["kind"]] = counts.get(note["kind"], 0) + 1
        if not notes:
            traj_lines.append(f"- `{name}`: —")
        else:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            traj_lines.append(f"- `{name}`: {summary}")
    scalar_lines = []
    for name in (
        "v3_proto_opsd_instrdialog_seed456",
        "v3_rand_ce_instrdialog_seed456",
        "v3h_proto_opsd_instrdialog_seed456",
    ):
        stats = train_scalars(name)
        scalar_lines.append(
            f"- `{name}`: CE {fmt(stats['ce'])}, OPSD {fmt(stats['opsd'])}, "
            f"rollout tokens {fmt(stats['rollout_tokens'])}, grad norm {fmt(stats['grad_norm'])}, "
            f"segments logged {stats['n']}"
        )
    timing = timing_rows()
    timing_md = md_table(["run", "start", "end", "elapsed_sec", "gpu_after"], timing) if timing else "_无新 run 计时_"
    baselines = baseline_inventory()
    base_md = "\n".join(f"- `{name}`" for name in baselines) if baselines else "- —"

    measured = sum(1 for row in rows if row.get("seen_avg_score") is not None)
    text = f"""# ABLATION v3 实验结果与待补表

数字只来自 `final_metrics.json`。缺测写 {DASH}。旧协议 `v3_` 使用官方 eval 遗忘选样（`eval_split_monitor`）。`v3h_` 使用训练集留出 5/50（`train_holdout`），四个方法共用同一划分。两套协议分行，不与旧 0.474/0.450 混成同一协议。

遗忘列保持原始定义（越小越好）。交互表最后一列只对遗忘类指标给出符号翻转后的“越大越好”值，图中遗忘也取该翻转。梯度余弦未在 8B 上测量（保留计算图会顶满显存），不写数值。

已填实测 run 数：{measured} / {len(rows)}。

## 11 指标：旧协议 seed 456（已有，不重跑）

{eleven_md(rows, lambda r: r["prefix"]=="v3" and r["seed"]==456)}

## 11 指标：旧协议 seed 789 / 42

{eleven_md(rows, lambda r: r["prefix"]=="v3" and r["seed"] in (789, 42))}

## 11 指标：v3h 训练集留出监控 seed 456

{eleven_md(rows, lambda r: r["prefix"]=="v3h")}

## E1 交互

交互 = proto_opsd − rand_opsd − proto_ce + rand_ce。proto 主效应与 OPSD 主效应是两条边的平均。缺任一格则整行交互为 {DASH}。

### seed 456 旧协议

{old_inter}

### seed 456 v3h（训练集留出监控）

{new_inter}

{chr(10).join(seed_rows)}

## E5 轨迹（来自各段 eval 的 per_segment_accuracy）

{chr(10).join(traj_lines)}

## E8 真实 case（eval_debug 末段，各取一条 match 与一条不 match）

{chr(10).join(case_lines)}

## E4 训练标量（metrics.jsonl 均值）

温度保持 `rollout_temperature: 1.0`，JSD 没有单独蒸馏温度。梯度余弦未测。

{chr(10).join(scalar_lines)}

## E3

顺序 SFT 与段内交替目标已有 v1 run，协议与本表 v3 / v3h 不同，这里只作对照说明，不重训。

## E7 仓库内真实 baseline 入口

{base_md}

官方第三方树未纳入本仓库。

## E12 实测时长与显存

{timing_md}

## 待补（本轮不跑）

- E2：replay 槽 15/5 与 5/15。Pending。
- E6：replay 比例网格。Pending。
- E9–E11：Pending。
"""
    (DOCS / "ABLATION_V3_实验结果与待补表.md").write_text(text, encoding="utf-8")

    queue = """# ABLATION v3 Run Queue

单卡顺序。已有 `final_metrics.json` 且配置指纹一致则跳过。目录已存在但不完整则拒绝覆盖，续跑命令在 `/root/autodl-tmp/rp_lora_v3_runs/logs/<run>.resume_cmd`。

结果目录：`/root/autodl-tmp/rp_lora_v3_runs/runs`。

## 冒烟

- `v3h_proto_opsd_instrdialog_seed456_smoke2seg`：PGDR+CE+OPSD，InstrDialog，`max_segments: 2`，seed 456，`train_holdout`。

## 旧协议 12（eval_split_monitor，seed 789 与 42）

`v3_rand_ce`、`v3_proto_ce`、`v3_rand_opsd` × InstrDialog / InstrDialog++。不重跑 seed 456 因子实验，不重跑已有 `v3_proto_opsd` 的 456/789/42。

## v3h 8（train_holdout，seed 456）

`v3h_{rand_ce,proto_ce,rand_opsd,proto_opsd}` × 两个数据集。冒烟若泄漏则本段不跑。

## 不做

不重跑 CITB / LCIA / v0 / v1 / v2 / Lit。
"""
    (DOCS / "ABLATION_V3_Run_Queue.md").write_text(queue, encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    rows = collect_grid()
    write_csv(rows)
    write_docs(rows)
    plot_bars("v3", 456, "seen_avg_score", "Seen-Avg Acc", FIGS / "old_seed456_acc.png")
    plot_bars(
        "v3",
        456,
        "forgetting",
        "−Forgetting (larger better)",
        FIGS / "old_seed456_forgetting_flipped.png",
        negate=True,
    )
    plot_bars("v3h", 456, "seen_avg_score", "Seen-Avg Acc", FIGS / "v3h_seed456_acc.png")
    plot_bars(
        "v3h",
        456,
        "forgetting",
        "−Forgetting (larger better)",
        FIGS / "v3h_seed456_forgetting_flipped.png",
        negate=True,
    )
    print(f"rows={len(rows)} measured={sum(1 for r in rows if r['seen_avg_score'] is not None)}")


if __name__ == "__main__":
    main()
