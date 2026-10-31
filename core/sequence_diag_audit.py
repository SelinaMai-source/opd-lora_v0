"""
Step 2.5: audit sequence diagnostics (rank convention, failure table, margin aggregates, position logic).

Run after an overfit-8 diagnostic run (from repo root):

  python core/sequence_diag_audit.py /path/to/results/runs/<run_id>/debug/overfit8

Omit the path to use `results/runs/baseline_alignment_overfit/debug/overfit8`.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _parse_bool_cell(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes")


def verify_rank_equals_consistency(rows: List[Dict[str, str]]) -> Tuple[bool, List[Tuple[str, str, int, bool]]]:
    bad: List[Tuple[str, str, int, bool]] = []
    for r in rows:
        rank = int(float(r["gold_token_rank"]))
        ge = _parse_bool_cell(r.get("gold_equals_greedy", "false"))
        if (rank == 0) != ge:
            bad.append((r.get("step", ""), r.get("sample_id", ""), rank, ge))
    return len(bad) == 0, bad


def aggregate_margin_by_step(margin_csv: Path) -> List[Dict[str, Any]]:
    by_step: Dict[int, List[Dict[str, str]]] = {}
    with margin_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_step.setdefault(int(row["step"]), []).append(row)
    out: List[Dict[str, Any]] = []
    for step in sorted(by_step.keys()):
        rs = by_step[step]
        ranks = [int(float(x["gold_token_rank"])) for x in rs]
        gp = [float(x["gold_prob"]) for x in rs]
        yp = [float(x["greedy_prob"]) for x in rs]
        out.append(
            {
                "step": step,
                "n_samples": len(rs),
                "count_gold_rank_le_1": sum(1 for x in ranks if x <= 1),
                "count_gold_rank_le_5": sum(1 for x in ranks if x <= 5),
                "count_gold_rank_gt_100": sum(1 for x in ranks if x > 100),
                "median_gold_rank": float(statistics.median(ranks)) if ranks else 0.0,
                "mean_gold_prob": float(sum(gp) / len(gp)) if gp else 0.0,
                "mean_greedy_prob": float(sum(yp) / len(yp)) if yp else 0.0,
            }
        )
    return out


def write_aggregate_csv(margin_csv: Path, out_csv: Path) -> None:
    rows = aggregate_margin_by_step(margin_csv)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_step25_report(
    *,
    overfit8_dir: Path,
    failure_step: int = 20,
) -> str:
    margin_csv = overfit8_dir / "first_token_margin_over_time.csv"
    failure_json = overfit8_dir / f"step_{failure_step:04d}_failure_analysis.json"

    lines: List[str] = [
        "# Step 2.5 — Diagnosis definition audit",
        "",
        f"- **overfit8_dir**: `{overfit8_dir}`",
        f"- **failure step**: {failure_step}",
        "",
        "## 1) `gold_token_rank` convention & `gold_equals_greedy` consistency",
        "",
        "**Rank is 0-based:** `0` = highest logit (greedy / top-1). Rank `k` means the gold first-token id appears at position `k` in logits sorted **descending** (tie-breaking follows `torch.argsort` order).",
        "",
        "**`gold_equals_greedy`:** `gold_token_id == greedy_token_id` where `greedy_token_id = argmax(logits at the audited position)`.",
        "",
        "Therefore **`gold_equals_greedy` is true iff `gold_token_rank == 0`** (strictly, barring implementation bugs). Below: all rows in `first_token_margin_over_time.csv` checked.",
        "",
    ]

    if margin_csv.is_file():
        all_rows = list(csv.DictReader(margin_csv.open(encoding="utf-8")))
        ok, bad = verify_rank_equals_consistency(all_rows)
        lines.append(f"- **Full CSV consistency** (rank==0 ⇔ gold_equals_greedy): **{'PASS' if ok else 'FAIL'}**")
        if bad:
            lines.append(f"- Mismatches: `{bad[:20]}` …" if len(bad) > 20 else f"- Mismatches: `{bad}`")
        lines.append("")
        lines.append(f"### Step {failure_step} — `sample_id`, `gold_token_rank`, `gold_equals_greedy`")
        lines.append("")
        lines.append("| sample_id | gold_token_rank | gold_equals_greedy |")
        lines.append("|---:|---:|:---:|")
        for r in all_rows:
            if int(r["step"]) != failure_step:
                continue
            lines.append(
                f"| {r['sample_id']} | {r['gold_token_rank']} | {r['gold_equals_greedy']} |"
            )
        lines.append("")
    else:
        lines.append("_Missing `first_token_margin_over_time.csv`._")
        lines.append("")

    lines.extend(
        [
            "## 2) Failure records (readable)",
            "",
        ]
    )
    if failure_json.is_file():
        recs = json.loads(failure_json.read_text(encoding="utf-8"))
        for rec in recs:
            sid = rec.get("sample_id")
            lines.append(f"### sample_id = {sid}")
            lines.append("")
            for key in (
                "gold_text",
                "pred_text_raw",
                "pred_text_trimmed",
                "pred_text_normalized",
            ):
                val = rec.get(key, "")
                lines.append(f"- **{key}**:")
                lines.append("")
                lines.append("```")
                lines.append(str(val))
                lines.append("```")
                lines.append("")
            tags = rec.get("failure_tags", [])
            lines.append(f"- **failure_tags**: `{tags}`")
            lines.append("")
    else:
        lines.append(f"_Missing `{failure_json.name}`._")
        lines.append("")

    lines.extend(
        [
            "## 3) `first_token_margin_over_time.csv` — aggregates by step",
            "",
            "Companion file (generated next to the margin CSV): **`first_token_margin_aggregate_by_step.csv`**",
            "",
            "Columns:",
            "",
            "- `count_gold_rank_le_1` — rank ≤ 1 (top-2 under 0-based ranking)",
            "- `count_gold_rank_le_5` — rank ≤ 5",
            "- `count_gold_rank_gt_100`",
            "- `median_gold_rank`, `mean_gold_prob`, `mean_greedy_prob`",
            "",
        ]
    )

    if margin_csv.is_file():
        agg = aggregate_margin_by_step(margin_csv)
        lines.append("| step | n | rank≤1 | rank≤5 | rank>100 | median_rank | mean_p(gold) | mean_p(greedy) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for a in agg:
            lines.append(
                f"| {a['step']} | {a['n_samples']} | {a['count_gold_rank_le_1']} | {a['count_gold_rank_le_5']} | "
                f"{a['count_gold_rank_gt_100']} | {a['median_gold_rank']:.1f} | {a['mean_gold_prob']:.6f} | {a['mean_greedy_prob']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 4) First-token audit — position alignment (not off-by-one)",
            "",
            "### Which logits are used?",
            "",
            "For causal LMs, `outputs.logits[b, t, :]` predicts the **next** token after position `t`, i.e. the distribution for `input_ids[b, t+1]`.",
            "",
            "We take **`logits = out.logits[0, -1, :]`**, i.e. the vector at the **last prompt token index** `T-1` where `T = len(input_ids)`.",
            "That is exactly the distribution over the **first newly generated token** — the same decision the model makes on **step 1** of `generate()` after consuming the full `infer_prompt` (including `add_generation_prompt=True` so the cursor sits just before assistant content).",
            "",
            "### Match to `gold` first token",
            "",
            "`gold_first_id = tokenizer.encode(gold_output, add_special_tokens=False)[0]` is the **first byte-pair / SentencePiece token** of the reference answer string, same convention as the continuation after the prompt in training data.",
            "",
            "### Why this is not off-by-one vs generation",
            "",
            "Hugging Face `model.generate` appends tokens starting from the distribution conditioned on **all** of `input_ids`. The first appended token is sampled/argmax from **the same** next-token distribution as `logits[0, -1]`. There is no extra shift between our audit slice and the first generation step.",
            "",
            "### Caveat (definition, not indexing bug)",
            "",
            "If the chat template emits a **leading space** as its own first token (e.g. token id for `\" \"` before `\"Hahah\"`), then `gold_first_id` must be that space token to match generation. We use raw `encode(gold)` of the dataset string; if the dataset omits a leading space that the template implicitly expects, rank can look worse without an off-by-one error in `logits[-1]`.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    ap = argparse.ArgumentParser(description="Step 2.5 sequence diagnosis audit")
    ap.add_argument(
        "overfit8_dir",
        nargs="?",
        default=str(repo / "results/runs/baseline_alignment_overfit/debug/overfit8"),
        help="Path to .../debug/overfit8",
    )
    ap.add_argument("--failure-step", type=int, default=20)
    args = ap.parse_args()
    od = Path(args.overfit8_dir)

    report = build_step25_report(overfit8_dir=od, failure_step=args.failure_step)
    out_md = od / "STEP25_DIAGNOSIS_AUDIT.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"Wrote {out_md}")

    margin = od / "first_token_margin_over_time.csv"
    if margin.is_file():
        agg_path = od / "first_token_margin_aggregate_by_step.csv"
        write_aggregate_csv(margin, agg_path)
        print(f"Wrote {agg_path}")


if __name__ == "__main__":
    main()
