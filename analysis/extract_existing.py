#!/usr/bin/env python3
"""CPU-only inventory of existing v3 / Lit / CITB artifacts.

Reads each run's own config_snapshot.yaml, final_metrics.json, metrics.jsonl,
replay_log.jsonl, segment dumps, and final eval_debug. Does not train.
"""

from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/rp_lora_v0")
RUNS = ROOT / "results" / "runs"
BENCH = ROOT / "benchmark"
LOGS = ROOT / "logs"
OUT = Path("/root/autodl-tmp/rp_lora_v3_analysis_draft")

ELEVEN = [
    ("seen_avg_score", "Seen-Avg Acc", True),
    ("seen_avg_task_aware_score", "Seen-Avg Task-aware Acc", True),
    ("forgetting", "Forgetting", False),
    ("task_aware_forgetting", "Task-Aware Forgetting", False),
    ("token_f1_mean", "Token F1", True),
    ("rouge_l_mean", "ROUGE-L", True),
    ("bleu_mean", "BLEU", True),
    ("lcs_overlap_mean", "LCS Overlap", True),
    ("current_score", "Current-Seg Acc", True),
    ("current_task_aware_score", "Current-Seg Task-aware Acc", True),
    ("task_aware_score_mean", "Task-aware Score Mean", True),
]

FACTORIAL_SEED456 = {
    ("instrdialog", "random_ce"): "v3_rand_ce_instrdialog_seed456",
    ("instrdialog", "pgdr_ce"): "v3_proto_ce_instrdialog_seed456",
    ("instrdialog", "random_opsd"): "v3_rand_opsd_instrdialog_seed456",
    ("instrdialog", "pgdr_opsd"): "v3_proto_opsd_instrdialog_seed456",
    ("instrdialogpp", "random_ce"): "v3_rand_ce_instrdialogpp_seed456",
    ("instrdialogpp", "pgdr_ce"): "v3_proto_ce_instrdialogpp_seed456",
    ("instrdialogpp", "random_opsd"): "v3_rand_opsd_instrdialogpp_seed456",
    ("instrdialogpp", "pgdr_opsd"): "v3_proto_opsd_instrdialogpp_seed456",
}

PROTO_SEEDS = {
    ("instrdialog", 456): "v3_proto_opsd_instrdialog_seed456",
    ("instrdialog", 789): "v3_proto_opsd_instrdialog_seed789",
    ("instrdialog", 42): "v3_proto_opsd_instrdialog_seed42",
    ("instrdialogpp", 456): "v3_proto_opsd_instrdialogpp_seed456",
    ("instrdialogpp", 789): "v3_proto_opsd_instrdialogpp_seed789",
    ("instrdialogpp", 42): "v3_proto_opsd_instrdialogpp_seed42",
}

REF_FACTORIAL = {
    ("instrdialog", "random_ce"): (0.395, 0.072),
    ("instrdialog", "pgdr_ce"): (0.384, 0.050),
    ("instrdialog", "random_opsd"): (0.400, 0.044),
    ("instrdialog", "pgdr_opsd"): (0.432, 0.017),
    ("instrdialogpp", "random_ce"): (0.376, 0.130),
    ("instrdialogpp", "pgdr_ce"): (0.474, 0.035),
    ("instrdialogpp", "random_opsd"): (0.389, 0.116),
    ("instrdialogpp", "pgdr_opsd"): (0.450, 0.068),
}
REF_PROTO = {
    ("instrdialog", 456): (0.432, 0.017),
    ("instrdialog", 789): (0.384, 0.050),
    ("instrdialog", 42): (0.368, 0.067),
    ("instrdialogpp", 456): (0.450, 0.068),
    ("instrdialogpp", 789): (0.455, 0.049),
    ("instrdialogpp", 42): (0.442, 0.073),
}

MISSING = object()


def parse_scalar(text: str):
    s = text.strip()
    if s in {"", "null", "~", "Null", "NULL"}:
        return None
    if s in {"true", "True", "yes"}:
        return True
    if s in {"false", "False", "no"}:
        return False
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    try:
        if any(ch in s for ch in ".eE"):
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_yaml(text: str):
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        lines.append(raw.rstrip())
    root = {}
    stack = [(-1, root)]
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while stack:
            open_indent, container = stack[-1]
            if indent > open_indent:
                break
            if (
                indent == open_indent
                and content.startswith("- ")
                and isinstance(container, list)
            ):
                break
            stack.pop()
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"list item outside list: {content}")
            parent.append(parse_scalar(content[2:]))
            continue
        key, sep, rest = content.partition(":")
        if not sep:
            raise ValueError(f"no colon: {content}")
        key = key.strip()
        rest = rest.strip()
        if rest == "":
            child = {}
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt.startswith("- "):
                    child = []
            if not isinstance(parent, dict):
                raise ValueError(f"nested key under non-dict: {key}")
            parent[key] = child
            stack.append((indent, child))
        else:
            if not isinstance(parent, dict):
                raise ValueError(f"scalar under non-dict: {key}")
            parent[key] = parse_scalar(rest)
    return root


def dig(obj, *path):
    cur = obj
    for part in path:
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def csv_cell(value) -> str:
    if value is MISSING or value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, list):
        return "|".join(str(x) for x in value)
    return str(value)


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_cell(row.get(k)) for k in fieldnames})


def family_of(name: str) -> str:
    if name.startswith("v3_"):
        return "v3"
    if name.startswith("lit_"):
        return "lit"
    if name.startswith("v2pre_citb_"):
        return "citb"
    if name.startswith("v2pre_lcia_"):
        return "lcia"
    return "null"


def benchmark_of(cfg: dict, name: str):
    stream_dir = dig(cfg, "paths", "processed_stream_dir")
    if stream_dir is MISSING or stream_dir is None:
        return None
    text = str(stream_dir).replace("\\", "/")
    if text.endswith("instrdialogpp") or "/instrdialogpp" in text:
        return "instrdialogpp"
    if text.endswith("instrdialog") or "/instrdialog" in text:
        return "instrdialog"
    return None


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def parse_dt(text: str):
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def elapsed_seconds(start, end):
    if not start or not end:
        return None
    try:
        return (parse_dt(end) - parse_dt(start)).total_seconds()
    except Exception:
        return None


def eleven_from_final(final: dict) -> dict:
    out = {}
    for key, _label, _hib in ELEVEN:
        field = f"eval.{key}"
        out[key] = final[field] if field in final else None
    return out


def round3(value):
    if value is None:
        return None
    return float(f"{float(value):.3f}")


def method_label(cfg: dict) -> str:
    baseline = dig(cfg, "baseline_name")
    if baseline is MISSING:
        return None
    parts = [str(baseline)]
    strategy = dig(cfg, "replay", "strategy")
    if strategy is not MISSING:
        parts.append(f"strategy={strategy}")
    lam = dig(cfg, "ce_opsd", "lambda_opsd")
    if lam is not MISSING:
        parts.append(f"lambda={lam}")
    slot = dig(cfg, "replay", "transfer_slot")
    if slot is not MISSING:
        parts.append(f"slot={slot}")
    cov = dig(cfg, "replay", "coverage_dedup")
    if cov is not MISSING:
        parts.append(f"coverage_dedup={cov}")
    mu = dig(cfg, "metric_opsd", "mu")
    if mu is not MISSING:
        parts.append(f"mu={mu}")
    enabled = dig(cfg, "replay", "enabled")
    if enabled is not MISSING:
        parts.append(f"replay_enabled={enabled}")
    single = dig(cfg, "debug_tools", "enable_single_segment_mode")
    if single is True:
        idx = dig(cfg, "debug_tools", "single_segment_index")
        parts.append(f"single_segment={idx if idx is not MISSING else 'null'}")
    return "|".join(parts)


def memory_capacity(cfg: dict):
    for path in (
        ("replay", "buffer_k"),
        ("replay", "k"),
        ("citb", "memory_k"),
        ("citb", "k"),
    ):
        value = dig(cfg, *path)
        if value is not MISSING:
            return value
    return None


def last_rows_by_segment(rows):
    ordered = []
    index = {}
    for row in rows:
        sid = row.get("segment_id")
        if sid in index:
            ordered[index[sid]] = row
        else:
            index[sid] = len(ordered)
            ordered.append(row)
    return ordered


def sum_metric(rows, key):
    rows = last_rows_by_segment(rows)
    if not rows or not any(key in row for row in rows):
        return None
    total = 0.0
    for row in rows:
        if key in row and row[key] is not None:
            total += float(row[key])
    return total


def protocol_string(cfg: dict) -> str:
    if not cfg:
        return None
    fmt = dig(cfg, "eval_normalization", "format_normalize")
    stream = dig(cfg, "paths", "processed_stream_file")
    max_seg = dig(cfg, "data", "max_segments")
    debug_n = dig(cfg, "eval", "debug_max_examples")
    parts = [
        f"format_normalize={csv_cell(fmt)}",
        f"stream={csv_cell(stream)}",
        f"max_segments={csv_cell(max_seg)}",
        f"debug_max_examples={csv_cell(debug_n)}",
    ]
    return ";".join(parts)


def verification(name: str, cfg: dict, final: dict, metrics: dict) -> str:
    keys = set(final)
    measured = all(f"eval.{k}" in keys for k, _l, _h in ELEVEN)
    if not measured:
        return "placeholder_not_a_measurement"
    bits = ["measured"]
    if "smoke" in name or dig(cfg, "data", "max_segments") == 2:
        bits.append("smoke_or_max_segments_2")
    if dig(cfg, "debug_tools", "enable_single_segment_mode") is True:
        bits.append("single_segment")
    baseline = dig(cfg, "baseline_name")
    strategy = dig(cfg, "replay", "strategy")
    forgotten = dig(cfg, "replay", "forgotten_ratio")
    uses_v3_selector = baseline in {"ce_opsd_replay", "metric_opsd_replay"}
    if uses_v3_selector and strategy == "proto" and isinstance(forgotten, (int, float)) and float(forgotten) > 0:
        bits.append("eval_split_monitor")
    elif uses_v3_selector and strategy == "uniform":
        bits.append("uniform_replay_does_not_call_select_forgotten")
    elif uses_v3_selector:
        bits.append("degraded_slot_not_configured")
    stream = dig(cfg, "paths", "processed_stream_file")
    if isinstance(stream, str) and "multiref" not in stream:
        bits.append("stream_file_not_multiref")
    fmt = dig(cfg, "eval_normalization", "format_normalize")
    if fmt is MISSING:
        bits.append("format_normalize_key_absent")
    if metrics.get("seen_segments") is not None:
        bits.append(f"num_seen_segments={metrics['seen_segments']}")
    return ";".join(bits)


def run_dirs():
    found = []
    for path in sorted(RUNS.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if not (
            name.startswith("v3_")
            or name.startswith("lit_")
            or name.startswith("v2pre_citb_")
            or name.startswith("v2pre_lcia_")
        ):
            continue
        if (path / "final_metrics.json").is_file():
            found.append(path)
    return found


def stream_stats(rel: str) -> dict:
    path = BENCH / rel
    raw = load_json(path)
    segs = raw["stream"]
    eval_lens = [len(seg.get("eval", [])) for seg in segs]
    train_lens = [len(seg.get("train", [])) for seg in segs]
    multi = 0
    total = 0
    overlap = 0
    for seg in segs:
        train_keys = {(ex.get("instruction"), ex.get("input")) for ex in seg.get("train", [])}
        for ex in seg.get("eval", []):
            total += 1
            golds = ex.get("outputs")
            if isinstance(golds, list) and len([g for g in golds if str(g)]) > 1:
                multi += 1
            if (ex.get("instruction"), ex.get("input")) in train_keys:
                overlap += 1
    return {
        "file": rel,
        "n_segments": len(segs),
        "eval_len_min": min(eval_lens) if eval_lens else None,
        "eval_len_max": max(eval_lens) if eval_lens else None,
        "train_len_min": min(train_lens) if train_lens else None,
        "train_len_max": max(train_lens) if train_lens else None,
        "eval_examples": total,
        "eval_examples_with_multiple_outputs": multi,
        "eval_train_key_overlap": overlap,
    }


def matrix_from_segments(run_dir: Path):
    rows = []
    for path in sorted((run_dir).glob("segment_*/eval_metrics.json")):
        payload = load_json(path)
        extra = payload.get("extra") or {}
        stage = payload.get("num_seen_segments")
        stage_id = None
        acc_rows = extra.get("per_segment_accuracy") or []
        ta_rows = extra.get("per_segment_task_aware_accuracy") or []
        if acc_rows:
            stage_id = max(int(r["segment_id"]) for r in acc_rows)
        ta_map = {int(r["segment_id"]): float(r["task_aware_accuracy"]) for r in ta_rows}
        for row in acc_rows:
            sid = int(row["segment_id"])
            rows.append(
                {
                    "stage_id": stage_id if stage_id is not None else stage,
                    "task_id": sid,
                    "acc": float(row["accuracy"]),
                    "task_aware_acc": ta_map.get(sid),
                }
            )
    return rows


def classify(series):
    vals = [float(v) for v in series]
    if not vals:
        return "no_series"
    peak = max(vals)
    final = vals[-1]
    trough = min(vals)
    if peak == 0.0:
        return "never_learned"
    rose_after_drop = False
    running_max = vals[0]
    seen_drop = False
    for value in vals[1:]:
        if value < running_max - 1e-12:
            seen_drop = True
        if seen_drop and value > min(vals[: vals.index(value)]) + 1e-12 and value > vals[vals.index(value) - 1] + 1e-12:
            rose_after_drop = True
        running_max = max(running_max, value)
    if abs(final - peak) <= 1e-12 and trough + 1e-12 < peak:
        return "forgotten_then_recovered"
    if final + 1e-12 < peak:
        if rose_after_drop:
            return "learned_then_forgotten_with_partial_recovery"
        return "learned_then_forgotten"
    return "retained"


def classify_series(values):
    vals = [float(v) for v in values]
    if not vals:
        return "no_series"
    peak = max(vals)
    final = vals[-1]
    trough = min(vals)
    if peak <= 1e-12:
        return "never_learned"
    dropped = False
    recovered = False
    running_max = vals[0]
    worst_after_drop = None
    for value in vals[1:]:
        if value < running_max - 1e-12:
            dropped = True
            worst_after_drop = value if worst_after_drop is None else min(worst_after_drop, value)
        elif dropped and worst_after_drop is not None and value > worst_after_drop + 1e-12:
            recovered = True
        running_max = max(running_max, value)
    if not dropped:
        return "retained"
    if abs(final - peak) <= 1e-12:
        return "forgotten_then_recovered"
    if recovered:
        return "learned_then_forgotten_with_partial_recovery"
    return "learned_then_forgotten"


def selector_audit(run_dir: Path) -> dict:
    logs = last_rows_by_segment(load_jsonl(run_dir / "replay_log.jsonl"))
    checked = 0
    argmax_hit = 0
    fallback = 0
    mismatch = 0
    no_dump = 0
    dup_segments = 0
    replay_rows = 0
    sum_replay = 0
    sum_neighbor = 0
    sum_forgotten = 0
    sum_uniform = 0
    sum_unique_deficit = 0
    for row in logs:
        sid = int(row.get("segment_id", -1))
        n_replay = int(row.get("n_replay") or 0)
        n_forg = int(row.get("n_forgotten") or 0)
        replay_rows += 1
        sum_replay += n_replay
        sum_neighbor += int(row.get("n_neighbor") or 0)
        sum_forgotten += n_forg
        sum_uniform += int(row.get("n_uniform") or 0)
        unique_pairs = row.get("unique_pairs")
        if unique_pairs is not None and n_replay > int(unique_pairs):
            sum_unique_deficit += n_replay - int(unique_pairs)
            dup_segments += 1
        if n_forg <= 0 or sid <= 0:
            continue
        dump_path = run_dir / "segment_dumps" / f"segment_{sid - 1:03d}.json"
        if not dump_path.is_file():
            no_dump += 1
            continue
        dump = load_json(dump_path)
        forgetting = dump.get("forgetting_by_segment") or {}
        best_sid = None
        best_f = -1.0
        for key, value in forgetting.items():
            fv = float(value)
            if fv > best_f:
                best_f = fv
                best_sid = int(key)
        sources = []
        for item in row.get("per_segment") or []:
            if int(item.get("n_forgotten") or 0) > 0:
                sources.append(int(item.get("segment_id")))
        checked += 1
        if best_f <= 0:
            fallback += 1
        elif sources == [best_sid]:
            argmax_hit += 1
        else:
            mismatch += 1
    return {
        "replay_segments_logged": replay_rows,
        "sum_n_replay": sum_replay,
        "sum_n_neighbor": sum_neighbor,
        "sum_n_forgotten": sum_forgotten,
        "sum_n_uniform": sum_uniform,
        "segments_with_duplicate_pairs": dup_segments,
        "duplicate_pair_slots": sum_unique_deficit,
        "forgotten_segments_checked": checked,
        "forgotten_argmax_match": argmax_hit,
        "forgotten_nonpositive_fallback": fallback,
        "forgotten_source_mismatch": mismatch,
        "forgotten_missing_previous_dump": no_dump,
    }


def index_examples(payload):
    out = {}
    for ex in payload.get("examples") or []:
        key = (int(ex.get("source_segment_id", -1)), int(ex.get("source_example_idx", -1)))
        out[key] = ex
    return out


def clip_case(ex, run_id, checkpoint):
    return {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "benchmark": None,
        "segment_id": ex.get("source_segment_id"),
        "segment_name": ex.get("source_segment_name"),
        "example_idx": ex.get("source_example_idx"),
        "instruction": ex.get("instruction"),
        "input_text": ex.get("input_text"),
        "gold_output": ex.get("gold_output"),
        "gold_outputs": ex.get("gold_outputs"),
        "prediction": ex.get("raw_generated_output"),
        "normalized_prediction": ex.get("normalized_prediction"),
        "normalized_gold": ex.get("normalized_gold"),
        "strict_match": ex.get("strict_match"),
        "task_aware_match": ex.get("task_aware_match"),
        "task_aware_score": ex.get("task_aware_score"),
        "task_score_type": ex.get("task_score_type"),
        "token_f1": ex.get("token_f1"),
        "rouge_l": ex.get("rouge_l"),
        "bleu": ex.get("bleu"),
        "lcs_overlap": ex.get("lcs_overlap"),
        "seed": 456,
    }


def pick_cases(bundles):
    """bundles: list of (role, run_id, checkpoint, examples_by_key, last_sid)."""
    by_role = {role: (run_id, ckpt, idx, last_sid) for role, run_id, ckpt, idx, last_sid in bundles}
    roles = list(by_role)
    base_role = "pgdr_opsd" if "pgdr_opsd" in by_role else roles[0]
    keys = set(by_role[base_role][2])
    for role in roles:
        keys &= set(by_role[role][2])
    chosen = {}

    def row(role, key):
        run_id, ckpt, idx, _last = by_role[role]
        ex = idx[key]
        item = clip_case(ex, run_id, ckpt)
        item["method_role"] = role
        return item

    def strict(role, key):
        return bool(by_role[role][2][key].get("strict_match"))

    def aware(role, key):
        return bool(by_role[role][2][key].get("task_aware_match"))

    def old(key):
        return key[0] != by_role[base_role][3]

    def short_key(cands):
        def length(key):
            ex = by_role[base_role][2][key]
            return len(str(ex.get("instruction") or "")) + len(str(ex.get("gold_output") or ""))
        return min(cands, key=length)

    groups = {
        "historical_retention": [
            k for k in keys if old(k) and strict("pgdr_opsd", k)
        ],
        "pgdr_improvement_under_opsd": [
            k for k in keys if strict("pgdr_opsd", k) and not strict("random_opsd", k)
        ],
        "pgdr_improvement_under_ce": [
            k for k in keys if strict("pgdr_ce", k) and not strict("random_ce", k)
        ],
        "opsd_improvement_under_random": [
            k for k in keys if strict("random_opsd", k) and not strict("random_ce", k)
        ],
        "opsd_improvement_under_pgdr": [
            k for k in keys if strict("pgdr_opsd", k) and not strict("pgdr_ce", k)
        ],
        "opsd_failure_under_pgdr": [
            k for k in keys if old(k) and strict("pgdr_ce", k) and not strict("pgdr_opsd", k)
        ],
        "current_task_pgdr_opsd_correct": [
            k for k in keys if not old(k) and strict("pgdr_opsd", k)
        ],
        "strict_em_false_task_aware_true": [
            k for k in keys if (not strict("pgdr_opsd", k)) and aware("pgdr_opsd", k)
        ],
    }
    selected = {}
    for name, cands in groups.items():
        if not cands:
            selected[name] = None
            continue
        key = short_key(cands)
        selected[name] = {
            "n_candidates": len(cands),
            "rows": [row(role, key) for role in ("random_ce", "pgdr_ce", "random_opsd", "pgdr_opsd") if role in by_role],
        }
    return selected


def main():
    (OUT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "cases").mkdir(parents=True, exist_ok=True)
    (OUT / "results" / "trajectories").mkdir(parents=True, exist_ok=True)

    manifest = []
    metric_rows = []
    wall_rows = []
    diag_rows = []
    replay_rows = []
    traj_rows = []
    class_rows = []
    yaml_errors = []

    for run_dir in run_dirs():
        name = run_dir.name
        cfg_path = run_dir / "config_snapshot.yaml"
        cfg = {}
        if cfg_path.is_file():
            try:
                cfg = parse_yaml(cfg_path.read_text(encoding="utf-8"))
            except Exception as exc:
                yaml_errors.append({"run_id": name, "error": str(exc)})
                cfg = {}
        final_doc = load_json(run_dir / "final_metrics.json")
        final = final_doc.get("final") or {}
        manifest_doc = {}
        man_path = run_dir / "run_manifest.json"
        if man_path.is_file():
            manifest_doc = load_json(man_path)
        jsonl_raw = load_jsonl(run_dir / "metrics.jsonl")
        jsonl = last_rows_by_segment(jsonl_raw)
        duplicate_segments = len(jsonl_raw) - len({row.get("segment_id") for row in jsonl_raw}) if jsonl_raw else 0
        metrics = eleven_from_final(final)
        seen = final.get("eval.num_seen_segments")
        metrics_meta = {"seen_segments": seen}
        steps = sum_metric(jsonl, "train.batches")
        if steps is None:
            steps = sum_metric(jsonl, "train.train.n_steps")
        tokens = sum_metric(jsonl, "train.num_supervised_tokens")
        if tokens is None:
            tokens = sum_metric(jsonl, "train.num_total_tokens")
        ckpt = run_dir / "final_adapter" / "adapter_model.safetensors"
        pred = run_dir / "eval_debug"
        status = manifest_doc.get("status")
        if status is None:
            status = "final_metrics_present"
        ver = verification(name, cfg, final, metrics_meta)
        if duplicate_segments:
            ver += f";metrics_jsonl_repeated_segment_rows={duplicate_segments}"
        if ver.startswith("placeholder"):
            status = "placeholder"
        bench = benchmark_of(cfg, name)
        replay_ratio = dig(cfg, "replay", "replay_ratio")
        related = dig(cfg, "replay", "neighbor_ratio")
        degraded = dig(cfg, "replay", "forgotten_ratio")
        lam = dig(cfg, "ce_opsd", "lambda_opsd")
        row = {
            "run_id": final_doc.get("run_id") or name,
            "experiment_version": family_of(name),
            "benchmark": bench,
            "method": method_label(cfg),
            "seed": dig(cfg, "seed") if dig(cfg, "seed") is not MISSING else None,
            "model": dig(cfg, "model", "hf_model_name_or_path") if dig(cfg, "model", "hf_model_name_or_path") is not MISSING else None,
            "lora_rank": dig(cfg, "lora", "r") if dig(cfg, "lora", "r") is not MISSING else None,
            "lora_alpha": dig(cfg, "lora", "alpha") if dig(cfg, "lora", "alpha") is not MISSING else None,
            "target_modules": dig(cfg, "lora", "target_modules") if dig(cfg, "lora", "target_modules") is not MISSING else None,
            "learning_rate": dig(cfg, "train", "lr") if dig(cfg, "train", "lr") is not MISSING else None,
            "memory_capacity": memory_capacity(cfg),
            "replay_ratio": None if replay_ratio is MISSING else replay_ratio,
            "related_ratio": None if related is MISSING else related,
            "degraded_ratio": None if degraded is MISSING else degraded,
            "opsd_lambda": None if lam is MISSING else lam,
            "training_steps": None if steps is None else int(steps),
            "training_tokens": None if tokens is None else int(tokens),
            "evaluation_protocol": protocol_string(cfg),
            "config_path": str(cfg_path) if cfg_path.is_file() else None,
            "source_commit": manifest_doc.get("git_commit_sha"),
            "checkpoint_path": str(ckpt) if ckpt.is_file() else None,
            "metrics_path": str(run_dir / "final_metrics.json"),
            "prediction_path": str(pred) if pred.is_dir() and any(pred.glob("*.json")) else None,
            "run_status": status,
            "verification_status": ver,
        }
        if row["run_id"] != name:
            row["verification_status"] += f";dir_name={name}"
        manifest.append(row)
        metric_rows.append(
            {
                "run_id": name,
                "experiment_version": row["experiment_version"],
                "benchmark": bench,
                "method": row["method"],
                "seed": row["seed"],
                "num_seen_segments": seen,
                "verification_status": ver,
                **metrics,
            }
        )
        start = manifest_doc.get("start_time")
        end = manifest_doc.get("end_time")
        log_start = (LOGS / f"{name}.start").read_text(encoding="utf-8").strip() if (LOGS / f"{name}.start").is_file() else None
        log_end = (LOGS / f"{name}.end").read_text(encoding="utf-8").strip() if (LOGS / f"{name}.end").is_file() else None
        wall_rows.append(
            {
                "run_id": name,
                "manifest_start": start,
                "manifest_end": end,
                "manifest_elapsed_seconds": elapsed_seconds(start, end),
                "log_start": log_start,
                "log_end": log_end,
                "log_elapsed_seconds": elapsed_seconds(log_start, log_end),
                "git_commit_sha": manifest_doc.get("git_commit_sha"),
                "git_dirty_count": len(manifest_doc.get("git_status_porcelain") or []) if manifest_doc else None,
            }
        )
        in_core = name in set(FACTORIAL_SEED456.values()) or name in set(PROTO_SEEDS.values())
        in_traj = in_core or (
            name.startswith("v2pre_citb_sequential_") and "smoke" not in name
        )
        if in_core:
            for mrow in jsonl:
                diag_rows.append(
                    {
                        "run_id": name,
                        "segment_id": mrow.get("segment_id"),
                        "train_ce": mrow.get("train.train.ce"),
                        "train_opsd": mrow.get("train.train.opsd"),
                        "train_loss": mrow.get("train.train.loss"),
                        "rollout_mean_tokens": mrow.get("train.rollout_mean_tokens"),
                        "grad_norm": mrow.get("train.grad_norm"),
                        "lambda_opsd": mrow.get("train.lambda_opsd"),
                        "opsd_applied": mrow.get("train.opsd_applied"),
                        "n_replay": mrow.get("train.n_replay"),
                        "n_neighbor": mrow.get("train.n_neighbor_replay"),
                        "n_forgotten": mrow.get("train.n_forgotten_replay"),
                        "n_uniform": mrow.get("train.n_uniform_replay"),
                        "seen_avg_score": mrow.get("eval.seen_avg_score"),
                        "current_score": mrow.get("eval.current_score"),
                        "forgetting": mrow.get("eval.forgetting"),
                    }
                )
        if (run_dir / "replay_log.jsonl").is_file() and name.startswith("v3_") and "smoke" not in name:
            audit = selector_audit(run_dir)
            audit["run_id"] = name
            replay_rows.append(audit)
        if in_traj:
            points = matrix_from_segments(run_dir)
            by_task = {}
            for point in points:
                traj_rows.append({"run_id": name, **point})
                by_task.setdefault(point["task_id"], []).append(point)
            for task_id, series_rows in sorted(by_task.items()):
                series_rows = sorted(series_rows, key=lambda r: int(r["stage_id"]))
                # A task is only scored once it has been seen; drop stages before task_id.
                series_rows = [r for r in series_rows if int(r["stage_id"]) >= int(task_id)]
                accs = [r["acc"] for r in series_rows]
                tas = [r["task_aware_acc"] for r in series_rows if r["task_aware_acc"] is not None]
                class_rows.append(
                    {
                        "run_id": name,
                        "task_id": task_id,
                        "n_stages": len(accs),
                        "acc_first": accs[0] if accs else None,
                        "acc_max": max(accs) if accs else None,
                        "acc_final": accs[-1] if accs else None,
                        "acc_class": classify_series(accs) if accs else None,
                        "task_aware_class": classify_series(tas) if tas else None,
                        "acc_series": " ".join(f"{v:.3f}" for v in accs),
                    }
                )

    manifest_fields = [
        "run_id",
        "experiment_version",
        "benchmark",
        "method",
        "seed",
        "model",
        "lora_rank",
        "lora_alpha",
        "target_modules",
        "learning_rate",
        "memory_capacity",
        "replay_ratio",
        "related_ratio",
        "degraded_ratio",
        "opsd_lambda",
        "training_steps",
        "training_tokens",
        "evaluation_protocol",
        "config_path",
        "source_commit",
        "checkpoint_path",
        "metrics_path",
        "prediction_path",
        "run_status",
        "verification_status",
    ]
    write_csv(OUT / "results" / "manifest.csv", manifest, manifest_fields)

    metric_fields = [
        "run_id",
        "experiment_version",
        "benchmark",
        "method",
        "seed",
        "num_seen_segments",
        "verification_status",
    ] + [k for k, _l, _h in ELEVEN]
    write_csv(OUT / "results" / "tables" / "all_scanned_eleven_metrics.csv", metric_rows, metric_fields)

    comparable = []
    citb_rows = []
    for row in metric_rows:
        ver = row["verification_status"] or ""
        if ver.startswith("placeholder") or "smoke" in ver or "single_segment" in ver:
            continue
        if any(row.get(k) is None for k, _l, _h in ELEVEN):
            continue
        proto = row["verification_status"]
        stream_note = "multiref" if "stream_file_not_multiref" not in ver and "format_normalize_key_absent" not in ver else "protocol_differs"
        if row["experiment_version"] in {"citb", "lcia"} or "stream_file_not_multiref" in ver or "format_normalize_key_absent" in ver:
            citb_rows.append({**row, "protocol_note": stream_note if row["experiment_version"] not in {"citb", "lcia"} else "citb_or_lcia_snapshot"})
        else:
            comparable.append(row)
    write_csv(
        OUT / "results" / "tables" / "eleven_metrics_comparable.csv",
        comparable,
        metric_fields,
    )
    write_csv(
        OUT / "results" / "tables" / "eleven_metrics_citb_lcia.csv",
        citb_rows,
        metric_fields + ["protocol_note"],
    )

    by_id = {row["run_id"]: row for row in metric_rows}
    interaction_rows = []
    for bench in ("instrdialog", "instrdialogpp"):
        cells = {}
        for role in ("random_ce", "pgdr_ce", "random_opsd", "pgdr_opsd"):
            rid = FACTORIAL_SEED456[(bench, role)]
            cells[role] = by_id[rid]
        for key, label, higher in ELEVEN:
            y00 = cells["random_ce"][key]
            y10 = cells["pgdr_ce"][key]
            y01 = cells["random_opsd"][key]
            y11 = cells["pgdr_opsd"][key]
            interaction = y11 - y10 - y01 + y00
            pgdr_effect = 0.5 * ((y11 - y01) + (y10 - y00))
            opsd_effect = 0.5 * ((y11 - y10) + (y01 - y00))
            interaction_rows.append(
                {
                    "benchmark": bench,
                    "seed": 456,
                    "metric": key,
                    "metric_label": label,
                    "raw_forgetting_lower_is_better": (not higher),
                    "Y_random_ce": y00,
                    "Y_pgdr_ce": y10,
                    "Y_random_opsd": y01,
                    "Y_pgdr_opsd": y11,
                    "pgdr_main_effect": pgdr_effect,
                    "opsd_main_effect": opsd_effect,
                    "interaction_raw": interaction,
                    "interaction_higher_is_better": interaction if higher else -interaction,
                }
            )
    write_csv(
        OUT / "results" / "tables" / "factorial_interaction_seed456.csv",
        interaction_rows,
        [
            "benchmark",
            "seed",
            "metric",
            "metric_label",
            "raw_forgetting_lower_is_better",
            "Y_random_ce",
            "Y_pgdr_ce",
            "Y_random_opsd",
            "Y_pgdr_opsd",
            "pgdr_main_effect",
            "opsd_main_effect",
            "interaction_raw",
            "interaction_higher_is_better",
        ],
    )

    seed_rows = []
    for (bench, seed), rid in PROTO_SEEDS.items():
        src = by_id[rid]
        seed_rows.append({"run_id": rid, "benchmark": bench, "seed": seed, **{k: src[k] for k, _l, _h in ELEVEN}})
    write_csv(
        OUT / "results" / "tables" / "proto_opsd_seeds.csv",
        seed_rows,
        ["run_id", "benchmark", "seed"] + [k for k, _l, _h in ELEVEN],
    )

    alloc_ids = [
        "v3_rand_opsd_instrdialog_seed456",
        "v3_transfer_all_neighbor_instrdialog_seed456",
        "v3_proto_opsd_instrdialog_seed456",
        "v3_transfer_all_forgotten_instrdialog_seed456",
        "v3_rand_opsd_instrdialogpp_seed456",
        "v3_transfer_all_neighbor_instrdialogpp_seed456",
        "v3_proto_opsd_instrdialogpp_seed456",
        "v3_transfer_all_forgotten_instrdialogpp_seed456",
    ]
    alloc_rows = [by_id[rid] for rid in alloc_ids if rid in by_id]
    write_csv(OUT / "results" / "tables" / "allocation_existing_seed456.csv", alloc_rows, metric_fields)

    write_csv(
        OUT / "results" / "tables" / "wall_clock.csv",
        wall_rows,
        [
            "run_id",
            "manifest_start",
            "manifest_end",
            "manifest_elapsed_seconds",
            "log_start",
            "log_end",
            "log_elapsed_seconds",
            "git_commit_sha",
            "git_dirty_count",
        ],
    )
    write_csv(
        OUT / "results" / "trajectories" / "segment_accuracy_long.csv",
        traj_rows,
        ["run_id", "stage_id", "task_id", "acc", "task_aware_acc"],
    )
    write_csv(
        OUT / "results" / "trajectories" / "task_classes.csv",
        class_rows,
        [
            "run_id",
            "task_id",
            "n_stages",
            "acc_first",
            "acc_max",
            "acc_final",
            "acc_class",
            "task_aware_class",
            "acc_series",
        ],
    )
    write_csv(
        OUT / "results" / "trajectories" / "train_diag_factorial.csv",
        diag_rows,
        [
            "run_id",
            "segment_id",
            "train_ce",
            "train_opsd",
            "train_loss",
            "rollout_mean_tokens",
            "grad_norm",
            "lambda_opsd",
            "opsd_applied",
            "n_replay",
            "n_neighbor",
            "n_forgotten",
            "n_uniform",
            "seen_avg_score",
            "current_score",
            "forgetting",
        ],
    )
    write_csv(
        OUT / "results" / "tables" / "replay_selector_audit.csv",
        replay_rows,
        [
            "run_id",
            "replay_segments_logged",
            "sum_n_replay",
            "sum_n_neighbor",
            "sum_n_forgotten",
            "sum_n_uniform",
            "segments_with_duplicate_pairs",
            "duplicate_pair_slots",
            "forgotten_segments_checked",
            "forgotten_argmax_match",
            "forgotten_nonpositive_fallback",
            "forgotten_source_mismatch",
            "forgotten_missing_previous_dump",
        ],
    )

    ref_checks = []
    for key, rid in FACTORIAL_SEED456.items():
        acc = by_id[rid]["seen_avg_score"]
        forg = by_id[rid]["forgetting"]
        exp_a, exp_f = REF_FACTORIAL[key]
        ref_checks.append(
            {
                "kind": "factorial_seed456",
                "benchmark": key[0],
                "cell": key[1],
                "run_id": rid,
                "acc": acc,
                "acc_3dp": round3(acc),
                "ref_acc": exp_a,
                "acc_match_3dp": round3(acc) == exp_a,
                "forgetting": forg,
                "forgetting_3dp": round3(forg),
                "ref_forgetting": exp_f,
                "forgetting_match_3dp": round3(forg) == exp_f,
            }
        )
    for key, rid in PROTO_SEEDS.items():
        acc = by_id[rid]["seen_avg_score"]
        forg = by_id[rid]["forgetting"]
        exp_a, exp_f = REF_PROTO[key]
        ref_checks.append(
            {
                "kind": "proto_opsd_seed",
                "benchmark": key[0],
                "cell": str(key[1]),
                "run_id": rid,
                "acc": acc,
                "acc_3dp": round3(acc),
                "ref_acc": exp_a,
                "acc_match_3dp": round3(acc) == exp_a,
                "forgetting": forg,
                "forgetting_3dp": round3(forg),
                "ref_forgetting": exp_f,
                "forgetting_match_3dp": round3(forg) == exp_f,
            }
        )

    files = {
        "instrdialog_multiref": "instrdialog/citb_cl_dialogue_tasks_train50_eval10_multiref.json",
        "instrdialog_citb": "instrdialog/citb_cl_dialogue_tasks_train50_eval10.json",
        "instrdialogpp_multiref": "instrdialogpp/citb_cl_38_random_tasks_train50_eval10_multiref.json",
        "instrdialogpp_citb": "instrdialogpp/citb_cl_38_random_tasks_train50_eval10.json",
    }
    streams = {name: stream_stats(rel) for name, rel in files.items()}

    cases = {}
    for bench in ("instrdialog", "instrdialogpp"):
        bundles = []
        for role in ("random_ce", "pgdr_ce", "random_opsd", "pgdr_opsd"):
            rid = FACTORIAL_SEED456[(bench, role)]
            debugs = sorted((RUNS / rid / "eval_debug").glob("eval_segment_*.json"))
            payload = load_json(debugs[-1])
            ckpt = str(RUNS / rid / "final_adapter")
            bundles.append((role, rid, ckpt, index_examples(payload), int(payload["segment_id"])))
        cases[bench] = pick_cases(bundles)
        for _name, block in cases[bench].items():
            if not block:
                continue
            for item in block["rows"]:
                item["benchmark"] = "InstrDialog++" if bench == "instrdialogpp" else "InstrDialog"

    class_counts = {}
    for row in class_rows:
        class_counts.setdefault(row["run_id"], {})
        class_counts[row["run_id"]][row["acc_class"]] = class_counts[row["run_id"]].get(row["acc_class"], 0) + 1

    diag_means = {}
    for row in diag_rows:
        bucket = diag_means.setdefault(
            row["run_id"],
            {"n": 0, "ce": 0.0, "opsd": 0.0, "roll": 0.0, "grad": 0.0, "opsd_n": 0, "roll_n": 0},
        )
        bucket["n"] += 1
        if row["train_ce"] is not None:
            bucket["ce"] += float(row["train_ce"])
        if row["opsd_applied"] is True and row["train_opsd"] is not None:
            bucket["opsd"] += float(row["train_opsd"])
            bucket["opsd_n"] += 1
        if row["rollout_mean_tokens"] is not None and row["opsd_applied"] is True:
            bucket["roll"] += float(row["rollout_mean_tokens"])
            bucket["roll_n"] += 1
        if row["grad_norm"] is not None:
            bucket["grad"] += float(row["grad_norm"])
    diag_summary = {}
    for rid, bucket in diag_means.items():
        diag_summary[rid] = {
            "segments": bucket["n"],
            "mean_train_ce": bucket["ce"] / bucket["n"] if bucket["n"] else None,
            "mean_train_opsd_when_applied": bucket["opsd"] / bucket["opsd_n"] if bucket["opsd_n"] else None,
            "mean_rollout_tokens_when_applied": bucket["roll"] / bucket["roll_n"] if bucket["roll_n"] else None,
            "mean_grad_norm": bucket["grad"] / bucket["n"] if bucket["n"] else None,
        }

    temps = {}
    for rid in list(FACTORIAL_SEED456.values()) + list(PROTO_SEEDS.values()):
        cfg_path = RUNS / rid / "config_snapshot.yaml"
        cfg = parse_yaml(cfg_path.read_text(encoding="utf-8"))
        temps[rid] = {
            "rollout_temperature": dig(cfg, "opsd", "rollout_temperature"),
            "beta": dig(cfg, "opsd", "beta"),
            "divergence": dig(cfg, "opsd", "divergence"),
            "teacher_format_constraint": dig(cfg, "opsd", "teacher_format_constraint"),
            "lambda_opsd": dig(cfg, "ce_opsd", "lambda_opsd"),
            "strategy": dig(cfg, "replay", "strategy"),
            "replay_ratio": dig(cfg, "replay", "replay_ratio"),
            "neighbor_ratio": dig(cfg, "replay", "neighbor_ratio"),
            "forgotten_ratio": dig(cfg, "replay", "forgotten_ratio"),
        }
        temps[rid] = {k: (None if v is MISSING else v) for k, v in temps[rid].items()}

    missing_ids = []
    for method in ("rand_ce", "proto_ce", "rand_opsd"):
        for bench in ("instrdialog", "instrdialogpp"):
            for seed in (789, 42):
                rid = f"v3_{method}_{bench}_seed{seed}"
                if not (RUNS / rid / "final_metrics.json").is_file():
                    missing_ids.append(rid)

    summary = {
        "n_manifest_rows": len(manifest),
        "n_comparable": len(comparable),
        "n_citb_lcia_table": len(citb_rows),
        "yaml_errors": yaml_errors,
        "reference_checks": ref_checks,
        "stream_stats": streams,
        "class_counts": class_counts,
        "diag_summary": diag_summary,
        "replay_audit": replay_rows,
        "temperature": temps,
        "missing_old_protocol_runs": missing_ids,
        "cases": cases,
        "interaction": interaction_rows,
    }
    (OUT / "results" / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "results" / "cases" / "selected_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": len(manifest),
        "comparable": len(comparable),
        "citb": len(citb_rows),
        "yaml_errors": yaml_errors,
        "missing": missing_ids,
        "ref_mismatch": [r for r in ref_checks if not (r["acc_match_3dp"] and r["forgetting_match_3dp"])],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
