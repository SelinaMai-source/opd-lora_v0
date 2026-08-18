"""
Overfit-8 sequence behavior diagnostics (failure decomposition, first-token audit,
prefix rollouts, decode ablation, ladder metrics) — baseline EM path unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.data import Example
from core.evaluate import _lcs_overlap, _normalize, _token_f1
from core.sequence_behavior_diag import (
    build_failure_record,
    diagnostic_normalize,
    diagnostic_trim_at_stops,
    failure_summary_row,
    summarize_failure_records,
)
from core.train_labels import build_supervised_labels, supervised_span_from_labels


def _append_csv(path: Path, row: Dict[str, Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def metrics_on_predictions(
    *,
    examples: Sequence[Example],
    preds: Sequence[str],
    infer_prompts: Sequence[str],
    normalization_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Baseline-style metrics on 8 samples (main _normalize EM + prefix + F1/LCS)."""
    exact = 0
    dn = 0
    p1 = p3 = p5 = 0
    f1s: List[float] = []
    lcss: List[float] = []
    for ex, pred, pr in zip(examples, preds, infer_prompts):
        n_pred = _normalize(pred or "", prompt=pr, cfg=normalization_cfg)
        n_gold = _normalize(ex.output or "", prompt=pr, cfg=normalization_cfg)
        if n_pred == n_gold:
            exact += 1
        if diagnostic_normalize(pred or "") == diagnostic_normalize(ex.output or ""):
            dn += 1
        pt = [t for t in n_pred.split() if t]
        gt = [t for t in n_gold.split() if t]
        p1 += int(pt[:1] == gt[:1])
        p3 += int(pt[:3] == gt[:3])
        p5 += int(pt[:5] == gt[:5])
        f1s.append(float(_token_f1(n_pred, n_gold)))
        lcss.append(float(_lcs_overlap(n_pred, n_gold)))
    n = max(1, len(examples))
    return {
        "exact_match_count": int(exact),
        "diagnostic_normalized_em_count": int(dn),
        "raw_em_count_strict": int(sum(1 for ex, p in zip(examples, preds) if (ex.output or "") == (p or ""))),
        "trimmed_em_count": int(
            sum(
                1
                for ex, p in zip(examples, preds)
                if diagnostic_trim_at_stops(p or "").strip() == diagnostic_trim_at_stops(ex.output or "").strip()
            )
        ),
        "prefix1_acc": float(p1 / n),
        "prefix3_acc": float(p3 / n),
        "prefix5_acc": float(p5 / n),
        "token_f1_mean": float(sum(f1s) / n),
        "lcs_overlap_mean": float(sum(lcss) / n),
    }


def run_failure_decomposition_step(
    *,
    step: int,
    out_dir: Path,
    examples: Sequence[Example],
    preds: Sequence[str],
    infer_prompts: Sequence[str],
    normalization_cfg: Dict[str, Any],
) -> None:
    records: List[Dict[str, Any]] = []
    for i, (ex, pred, pr) in enumerate(zip(examples, preds, infer_prompts)):
        norm_p = _normalize(pred or "", prompt=pr, cfg=normalization_cfg)
        norm_g = _normalize(ex.output or "", prompt=pr, cfg=normalization_cfg)
        records.append(
            build_failure_record(
                sample_id=i,
                prompt_text=pr,
                gold_text=ex.output or "",
                pred_text_raw=pred or "",
                main_norm_pred=norm_p,
                main_norm_gold=norm_g,
            )
        )
    summary = summarize_failure_records(records, step=step)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"step_{step:04d}_failure_analysis.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from core.utils import save_csv

    save_csv(str(out_dir / f"step_{step:04d}_failure_summary.csv"), [failure_summary_row(summary)])


def hf_first_token_audit_one(
    backbone: Any,
    infer_prompt: str,
    gold_output: str,
    supervised_gold: Optional[Dict[str, Any]],
    sample_id: int,
    gold_source: str,
) -> Dict[str, Any]:
    """
    First-token audit at the **first generation step** (open-loop), aligned with `generate()` step 1.

    - **Position:** `logits = model(...).logits[0, -1, :]` — distribution for the token **after** the last
      prompt token (same as the first sampled/argmax token in HF `generate` given this `input_ids`).
    - **Gold token:** selectable source:
      - `raw_text`: first token from `tokenizer.encode(gold_output, add_special_tokens=False)`
      - `supervised_span`: first token from the final training supervision span (`labels != -100`)
    - **gold_token_rank:** **0-based** index in logits sorted descending; **0 == top-1 (greedy choice)**.
      `gold_equals_greedy` is True iff `gold_token_rank == 0` (same as `gold_token_id == argmax(logits)`).
    """
    import torch

    tok = backbone.tokenizer
    model = backbone.model
    device = backbone.device
    enc = tok(
        infer_prompt,
        return_tensors="pt",
        # `infer_prompt` is already a chat-template string with special tokens included.
        add_special_tokens=False,
        truncation=True,
        max_length=int(backbone.cfg.max_seq_len),
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    src = str(gold_source or "supervised_span").strip().lower()
    if src == "supervised_span":
        if not supervised_gold or supervised_gold.get("first_supervised_token_id") is None:
            return {
                "sample_id": sample_id,
                "error": "empty_supervised_span",
            }
        gold_first_id = int(supervised_gold["first_supervised_token_id"])
        gold_token_text = str(supervised_gold.get("first_supervised_token_text", ""))
        gold_source_detail = "supervised_span"
    else:
        gold_ids = tok.encode(str(gold_output or ""), add_special_tokens=False)
        if not gold_ids:
            return {
                "sample_id": sample_id,
                "error": "empty_gold_tokenization",
            }
        gold_first_id = int(gold_ids[0])
        gold_token_text = tok.convert_ids_to_tokens([gold_first_id])[0]
        gold_source_detail = "raw_text"
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = out.logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    greedy_id = int(torch.argmax(logits).item())
    gold_logit = float(logits[gold_first_id].item())
    greedy_logit = float(logits[greedy_id].item())
    gold_prob = float(probs[gold_first_id].item())
    greedy_prob = float(probs[greedy_id].item())
    sorted_idx = torch.argsort(logits, descending=True)
    rank_tensor = (sorted_idx == gold_first_id).nonzero(as_tuple=True)[0]
    rank = int(rank_tensor[0].item()) if rank_tensor.numel() else 999999
    topk = min(5, logits.numel())
    top_p, top_i = torch.topk(probs, k=topk)
    top5_token_ids: List[int] = []
    top5_token_texts: List[str] = []
    top5_token_probs: List[float] = []
    for j in range(topk):
        tid = int(top_i[j].item())
        top5_token_ids.append(tid)
        top5_token_texts.append(tok.convert_ids_to_tokens([tid])[0])
        top5_token_probs.append(float(top_p[j].item()))
    if was_training:
        model.train()
    return {
        "sample_id": int(sample_id),
        "gold_source": gold_source_detail,
        "gold_token_id": gold_first_id,
        "gold_token_text": gold_token_text,
        "greedy_token_id": greedy_id,
        "greedy_token_text": tok.convert_ids_to_tokens([greedy_id])[0],
        "gold_token_rank": rank,
        "gold_logit": gold_logit,
        "greedy_logit": greedy_logit,
        "gold_prob": gold_prob,
        "greedy_prob": greedy_prob,
        "margin_logit_greedy_minus_gold": float(greedy_logit - gold_logit),
        "top5_token_ids": top5_token_ids,
        "top5_token_texts": top5_token_texts,
        "top5_token_probs": top5_token_probs,
        "gold_equals_greedy": bool(gold_first_id == greedy_id),
    }


def run_first_token_audit_step(
    *,
    step: int,
    out_dir: Path,
    examples: Sequence[Example],
    infer_prompts: Sequence[str],
    backbone: Any,
    pairs: Sequence[Tuple[str, str]],
    first_token_gold_source: str = "supervised_span",
) -> None:
    rows: List[Dict[str, Any]] = []
    source_cmp_rows: List[Dict[str, Any]] = []
    cfg = getattr(backbone, "cfg", None)
    if cfg is None:
        raise RuntimeError("backbone.cfg is required for supervised_span first-token audit")
    for i, (ex, pr, pair) in enumerate(zip(examples, infer_prompts, pairs)):
        ins, inp = pair
        supervised_enc = build_supervised_labels(
            backbone.tokenizer,
            str(ins),
            str(inp),
            str(ex.output or ""),
            max_len=int(cfg.max_seq_len),
            min_target_tokens=int(cfg.min_target_tokens_for_loss),
            mask_eos_token_in_labels=bool(cfg.mask_eos_token_in_labels),
            mask_all_special_tokens_in_labels=bool(cfg.mask_all_special_tokens_in_labels),
            labeling_mode=str(cfg.train_labeling_mode),
            completion_only_response_template=str(cfg.completion_only_response_template),
        )
        supervised_gold = supervised_span_from_labels(
            tokenizer=backbone.tokenizer,
            full_ids=supervised_enc.full_ids,
            labels=supervised_enc.labels,
            preview_tokens=5,
        )
        row = hf_first_token_audit_one(
            backbone,
            pr,
            ex.output or "",
            supervised_gold,
            i,
            first_token_gold_source,
        )
        row["step"] = int(step)
        row["first_token_gold_source"] = str(first_token_gold_source)
        row.update(
            {
                "supervised_first_token_position": supervised_gold.get("first_supervised_token_position"),
                "supervised_first_token_id": supervised_gold.get("first_supervised_token_id"),
                "supervised_first_token_text": supervised_gold.get("first_supervised_token_text"),
                "supervised_preview_token_ids": supervised_gold.get("first_supervised_preview_token_ids", []),
                "supervised_preview_token_texts": supervised_gold.get("first_supervised_preview_token_texts", []),
            }
        )
        rows.append(row)
        # Compare raw tokenization variants vs supervised span gold.
        tok = backbone.tokenizer
        raw = str(ex.output or "")
        raw_ls = raw.lstrip()
        variants = {
            "raw_text": tok.encode(raw, add_special_tokens=False),
            "lstrip_raw_text": tok.encode(raw_ls, add_special_tokens=False),
            "space_plus_lstrip_raw_text": tok.encode(" " + raw_ls, add_special_tokens=False),
        }
        cmp_row: Dict[str, Any] = {
            "step": int(step),
            "sample_id": int(i),
            "supervised_first_token_id": supervised_gold.get("first_supervised_token_id"),
            "supervised_first_token_text": supervised_gold.get("first_supervised_token_text", ""),
        }
        for name, ids in variants.items():
            first_id = int(ids[0]) if ids else None
            first_text = tok.convert_ids_to_tokens([first_id])[0] if first_id is not None else ""
            cmp_row[f"{name}_first_token_id"] = first_id
            cmp_row[f"{name}_first_token_text"] = first_text
            cmp_row[f"{name}_matches_supervised"] = bool(
                first_id is not None and first_id == supervised_gold.get("first_supervised_token_id")
            )
        source_cmp_rows.append(cmp_row)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"step_{step:04d}_first_token_audit.json").write_text(
        json.dumps({"step": step, "samples": rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"step_{step:04d}_first_token_gold_source_compare.json").write_text(
        json.dumps({"step": step, "samples": source_cmp_rows}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from core.utils import save_csv

    save_csv(str(out_dir / "first_token_gold_source_compare.csv"), source_cmp_rows)
    ft_fields = (
        "step",
        "sample_id",
        "gold_source",
        "first_token_gold_source",
        "gold_token_id",
        "gold_token_text",
        "greedy_token_id",
        "greedy_token_text",
        "gold_token_rank",
        "gold_logit",
        "greedy_logit",
        "gold_prob",
        "greedy_prob",
        "margin_logit_greedy_minus_gold",
        "top5_token_ids",
        "top5_token_texts",
        "top5_token_probs",
        "gold_equals_greedy",
        "supervised_first_token_position",
        "supervised_first_token_id",
        "supervised_first_token_text",
        "supervised_preview_token_ids",
        "supervised_preview_token_texts",
    )
    csv_path = out_dir / "first_token_margin_over_time.csv"
    for r in rows:
        if r.get("error"):
            continue
        _append_csv(
            csv_path,
            {
                "step": r["step"],
                "sample_id": r["sample_id"],
                "gold_source": r.get("gold_source", ""),
                "first_token_gold_source": r.get("first_token_gold_source", ""),
                "gold_token_id": r["gold_token_id"],
                "gold_token_text": r["gold_token_text"],
                "greedy_token_id": r["greedy_token_id"],
                "greedy_token_text": r["greedy_token_text"],
                "gold_token_rank": r["gold_token_rank"],
                "gold_logit": r["gold_logit"],
                "greedy_logit": r["greedy_logit"],
                "gold_prob": r["gold_prob"],
                "greedy_prob": r["greedy_prob"],
                "margin_logit_greedy_minus_gold": r["margin_logit_greedy_minus_gold"],
                "top5_token_ids": json.dumps(r["top5_token_ids"], ensure_ascii=False),
                "top5_token_texts": json.dumps(r["top5_token_texts"], ensure_ascii=False),
                "top5_token_probs": json.dumps(r["top5_token_probs"], ensure_ascii=False),
                "gold_equals_greedy": r["gold_equals_greedy"],
                "supervised_first_token_position": r.get("supervised_first_token_position"),
                "supervised_first_token_id": r.get("supervised_first_token_id"),
                "supervised_first_token_text": r.get("supervised_first_token_text", ""),
                "supervised_preview_token_ids": json.dumps(r.get("supervised_preview_token_ids", []), ensure_ascii=False),
                "supervised_preview_token_texts": json.dumps(
                    r.get("supervised_preview_token_texts", []), ensure_ascii=False
                ),
            },
            ft_fields,
        )


def run_prefix_rollout_step(
    *,
    step: int,
    out_dir: Path,
    examples: Sequence[Example],
    infer_prompts: Sequence[str],
    backbone: Any,
    max_new_tokens: int,
    normalization_cfg: Dict[str, Any],
) -> None:
    """
    For each nominal prefix length in {0,1,3,5}, append that many **gold answer** token ids (tokenizer,
    add_special_tokens=False), then generate. If the gold string has fewer than ``pl`` tokens, we **clamp**
    to ``min(pl, len(gold_ids))`` so we still run generation (full gold prefix) instead of recording an
    empty prediction (which previously skewed EM/prefix metrics for short references).
    """
    tok = backbone.tokenizer
    prefix_lens = (0, 1, 3, 5)
    sample_rows: List[Dict[str, Any]] = []
    for i, (ex, pr) in enumerate(zip(examples, infer_prompts)):
        gold = ex.output or ""
        gold_ids = tok.encode(gold, add_special_tokens=False)
        row: Dict[str, Any] = {"sample_id": i, "gold_output": gold}
        for pl in prefix_lens:
            key = f"rollout_prefix_{pl}"
            if pl == 0:
                forced: List[int] = []
            else:
                if not gold_ids:
                    row[key] = ""
                    row[f"{key}_error"] = "empty_gold_output"
                    continue
                take = min(pl, len(gold_ids))
                forced = list(gold_ids[:take])
                row[f"{key}_requested_len"] = pl
                row[f"{key}_effective_len"] = take
                if take < pl:
                    row[f"{key}_clamped_short_gold"] = True
            text = backbone.generate_with_forced_answer_prefix(
                pr, max_new_tokens=max_new_tokens, forced_answer_token_ids=forced, num_beams=None
            )
            row[key] = text
        sample_rows.append(row)

    def _preds_and_clamp_counts(pl: int) -> Tuple[List[str], int, int]:
        preds: List[str] = []
        n_clamped = 0
        n_empty_skip = 0
        for r in sample_rows:
            key = f"rollout_prefix_{pl}"
            if pl > 0:
                if r.get(f"{key}_error") == "empty_gold_output":
                    n_empty_skip += 1
                elif r.get(f"{key}_clamped_short_gold"):
                    n_clamped += 1
            preds.append(r.get(key, ""))
        return preds, n_clamped, n_empty_skip

    rollout_summaries: Dict[str, Any] = {}
    for pl in prefix_lens:
        preds_pl, n_clamped, n_empty_skip = _preds_and_clamp_counts(pl)
        m = metrics_on_predictions(
            examples=examples, preds=preds_pl, infer_prompts=infer_prompts, normalization_cfg=normalization_cfg
        )
        rollout_summaries[str(pl)] = {
            **m,
            "n_clamped_short_gold": int(n_clamped),
            "n_empty_gold_skipped": int(n_empty_skip),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"step_{step:04d}_prefix_rollout.json").write_text(
        json.dumps(
            {"step": step, "rollout_summaries": rollout_summaries, "samples": sample_rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    pr_fields = (
        "step",
        "rollout_prefix_len",
        "n_clamped_short_gold",
        "n_empty_gold_skipped",
        "raw_em_count",
        "normalized_em_count",
        "prefix1_acc",
        "prefix3_acc",
        "prefix5_acc",
        "token_f1_mean",
        "lcs_overlap_mean",
    )
    csv_path = out_dir / "prefix_rollout_summary.csv"
    for pl in prefix_lens:
        preds, n_clamped, n_empty_skip = _preds_and_clamp_counts(pl)
        m = metrics_on_predictions(
            examples=examples, preds=preds, infer_prompts=infer_prompts, normalization_cfg=normalization_cfg
        )
        _append_csv(
            csv_path,
            {
                "step": step,
                "rollout_prefix_len": pl,
                "n_clamped_short_gold": int(n_clamped),
                "n_empty_gold_skipped": int(n_empty_skip),
                "raw_em_count": m["raw_em_count_strict"],
                "normalized_em_count": m["diagnostic_normalized_em_count"],
                "prefix1_acc": m["prefix1_acc"],
                "prefix3_acc": m["prefix3_acc"],
                "prefix5_acc": m["prefix5_acc"],
                "token_f1_mean": m["token_f1_mean"],
                "lcs_overlap_mean": m["lcs_overlap_mean"],
            },
            pr_fields,
        )


def run_decode_ablation(
    *,
    out_dir: Path,
    examples: Sequence[Example],
    infer_prompts: Sequence[str],
    backbone: Any,
    max_new_tokens: int,
    normalization_cfg: Dict[str, Any],
) -> None:
    """
    Deterministic open-loop decode comparison: greedy (num_beams=1) vs beam search (2, 4).
    Uses ``do_sample=False`` so decoding has no randomness; pad/eos and max_new_tokens follow backbone config.
    Prompts must already include ``add_generation_prompt=True`` (caller responsibility).
    """
    variants = (("greedy", 1), ("beam2", 2), ("beam4", 4))
    rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    for name, beams in variants:
        preds = backbone.generate(
            list(infer_prompts),
            max_new_tokens=max_new_tokens,
            num_beams=beams,
            do_sample=False,
        )
        m = metrics_on_predictions(
            examples=examples, preds=preds, infer_prompts=infer_prompts, normalization_cfg=normalization_cfg
        )
        row = {
            "decode_name": name,
            "num_beams": beams,
            "do_sample": False,
            "raw_em_count": m["raw_em_count_strict"],
            "trimmed_em_count": m["trimmed_em_count"],
            "normalized_em_count": m["diagnostic_normalized_em_count"],
            "baseline_eval_normalize_em_count": m["exact_match_count"],
            "prefix1_acc": m["prefix1_acc"],
            "prefix3_acc": m["prefix3_acc"],
            "prefix5_acc": m["prefix5_acc"],
            "token_f1_mean": m["token_f1_mean"],
            "lcs_overlap_mean": m["lcs_overlap_mean"],
        }
        rows.append(row)
        csv_rows.append(
            {
                "decode_name": name,
                "num_beams": beams,
                "do_sample": False,
                "raw_em_count": m["raw_em_count_strict"],
                "trimmed_em_count": m["trimmed_em_count"],
                "normalized_em_count": m["diagnostic_normalized_em_count"],
                "prefix1_acc": m["prefix1_acc"],
                "prefix3_acc": m["prefix3_acc"],
                "prefix5_acc": m["prefix5_acc"],
                "token_f1_mean": m["token_f1_mean"],
                "lcs_overlap_mean": m["lcs_overlap_mean"],
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "decode_ablation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    from core.utils import save_csv

    save_csv(
        str(out_dir / "decode_ablation.csv"),
        csv_rows,
    )


def run_overfit_ladder(
    *,
    ladder_sizes: Tuple[int, ...],
    ladder_steps: int,
    pairs: Sequence[Tuple[str, str]],
    targets: Sequence[str],
    train_subset: Sequence[Example],
    infer_prompts: Sequence[str],
    backbone: Any,
    lora: Any,
    lr: float,
    overfit_batch_size: int,
    overfit_gen_max_new_tokens: int,
    normalization_cfg: Dict[str, Any],
    ladder_init_adapter_path: str,
    out_csv: Path,
    logger: Any,
) -> None:
    from baselines.basic_baselines.sequential_lora.method import _batch
    from core.debug_teacher_audit import mean_teacher_forced_over_examples

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ladder_rows: List[Dict[str, Any]] = []
    for n in ladder_sizes:
        if hasattr(lora, "load_adapter_checkpoint"):
            lora.load_adapter_checkpoint(ladder_init_adapter_path)
        pairs_n = list(pairs[:n])
        targets_n = list(targets[:n])
        final_train_loss = 0.0
        for _step in range(1, ladder_steps + 1):
            inner_loss = 0.0
            inner_n = 0
            for b_pairs, b_targets in _batch(pairs_n, targets_n, overfit_batch_size):
                out = backbone.fit_batch(b_pairs, b_targets, lr=lr)
                lora.step_adapter()
                inner_loss += float(out.get("train_loss", 0.0))
                inner_n += 1
            final_train_loss = inner_loss / max(1, inner_n)
        tf_loss_m, tf_acc_m, _n_tf = mean_teacher_forced_over_examples(
            backbone=backbone, pairs=list(pairs), targets=list(targets)
        )
        preds = backbone.generate(list(infer_prompts), max_new_tokens=overfit_gen_max_new_tokens)
        m = metrics_on_predictions(
            examples=train_subset,
            preds=preds,
            infer_prompts=infer_prompts,
            normalization_cfg=normalization_cfg,
        )
        ladder_rows.append(
            {
                "ladder_n": int(n),
                "ladder_steps": int(ladder_steps),
                "final_train_loss": float(final_train_loss),
                "teacher_forced_shifted_token_acc_mean": float(tf_acc_m),
                "raw_em_count": int(m["raw_em_count_strict"]),
                "normalized_em_count": int(m["diagnostic_normalized_em_count"]),
                "baseline_eval_normalize_em_count": int(m["exact_match_count"]),
                "prefix1_acc": m["prefix1_acc"],
                "prefix3_acc": m["prefix3_acc"],
                "prefix5_acc": m["prefix5_acc"],
                "token_f1_mean": m["token_f1_mean"],
                "lcs_overlap_mean": m["lcs_overlap_mean"],
            }
        )
        if logger is not None:
            logger.log(
                f"[overfit_ladder] n={n} final_loss={final_train_loss:.6f} tf_acc={tf_acc_m:.4f} "
                f"raw_EM={m['raw_em_count_strict']}/8 diag_norm_EM={m['diagnostic_normalized_em_count']}/8 "
                f"baseline_eval_norm_EM={m['exact_match_count']}/8"
            )
    from core.utils import save_csv

    save_csv(str(out_csv), ladder_rows)


def write_sequence_behavior_report(
    *,
    run_manifest_path: Path,
    results_dir: Path,
    experiment_name: str,
    run_id: str,
) -> None:
    """Aggregate diagnostics into a single markdown report using run manifest only."""
    manifest: Dict[str, Any] = {}
    if run_manifest_path.is_file():
        try:
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    artifacts = manifest.get("artifacts_generated_in_run", [])
    art_set = {str(p) for p in artifacts} if isinstance(artifacts, list) else set()

    def _manifest_path(name: str) -> Optional[Path]:
        for p in sorted(art_set):
            pp = Path(p)
            if pp.name == name:
                return pp
        return None

    overfit8 = run_manifest_path.parent
    lines: List[str] = [
        "# Sequence behavior diagnosis report",
        "",
        "## Framing",
        "",
        "Observed pattern: **train loss → ~0**, **shifted teacher-forced token accuracy → ~1**, "
        "**train-mode answer token acc → ~1**, while **greedy open-loop exact match** stays low (e.g. ~1/8) "
        "and **prefix** metrics lag. We treat this as: **conditional distributions under teacher forcing are "
        "well fit**, but the **autoregressive rollout** (same weights, same prompts) does not land on the "
        "reference strings. This report does **not** re-litigate mask/shift correctness; it aggregates "
        "artifacts that localize *where* the open-loop trajectory diverges (first token, prefix, surface form, "
        "decode path, or deeper sequence mismatch).",
        "",
        f"- **run_id**: `{run_id}`",
        f"- **experiment**: `{experiment_name}`",
        f"- **artifacts**: `{overfit8}`",
        "",
        "Metrics legend: **baseline_eval_normalize_EM** uses `evaluate._normalize` (unchanged baseline). "
        "**strict_string_EM** is exact string equality. **diagnostic_norm_EM** uses `diagnostic_normalize` only.",
        "",
        "## Q1 — What kind of failures dominate?",
        "",
        "Compare `count_*` columns in the latest `step_XXXX_failure_summary.csv` with `raw_em_count` / "
        "`trimmed_em_count` / `normalized_em_count` to see whether errors are first-token, prefix drift, "
        "surface form, stop/over-gen, or content.",
        "",
    ]
    summaries = sorted(Path(p) for p in art_set if Path(p).name.endswith("_failure_summary.csv"))
    last_summary_row: Optional[Dict[str, str]] = None
    if summaries:
        last = summaries[-1]
        lines.append(f"- Latest summary: `{last.name}`")
        try:
            with last.open(encoding="utf-8") as f:
                r = csv.DictReader(f)
                last_summary_row = next(iter(r), None)
            if last_summary_row:
                lines.append("")
                lines.append("| Metric | Value |")
                lines.append("|---|---|")
                for k in sorted(last_summary_row.keys()):
                    if k.startswith("count_") or k.endswith("_em_count") or k in ("step", "num_samples"):
                        lines.append(f"| {k} | {last_summary_row[k]} |")
        except Exception as e:
            lines.append(f"(parse error: {e})")
    else:
        lines.append("_No failure summary CSVs found._")

    q1_answer = ""
    if last_summary_row:
        counts = []
        for k, v in last_summary_row.items():
            if k.startswith("count_"):
                try:
                    counts.append((k, int(float(v))))
                except ValueError:
                    pass
        if counts:
            counts.sort(key=lambda x: -x[1])
            topn = ", ".join(f"`{a}`={b}" for a, b in counts[:3])
            q1_answer = f"Dominant tagged failure modes (top 3): {topn}. Cross-check `raw_em_count` vs `trimmed_em_count` vs `normalized_em_count` for surface vs stop vs whitespace."
    lines.extend(["", "### Q1 answer (from latest failure summary)", "", q1_answer or "_No summary row._", ""])

    lines.extend(["", "## Q2 — First-token margin: rank, prob gap, logit margin", ""])
    ft_csv = _manifest_path("first_token_margin_over_time.csv")
    ranks: List[int] = []
    margins: List[float] = []
    gold_probs: List[float] = []
    greedy_probs: List[float] = []
    if ft_csv is not None and ft_csv.is_file():
        with ft_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ranks.append(int(float(row.get("gold_token_rank", "999"))))
                    margins.append(float(row.get("margin_logit_greedy_minus_gold", "0")))
                    gp = row.get("gold_prob", row.get("prob_gold", "0"))
                    yp = row.get("greedy_prob", row.get("prob_greedy", "0"))
                    gold_probs.append(float(gp))
                    greedy_probs.append(float(yp))
                except ValueError:
                    continue
        if ranks:
            avg_r = sum(ranks) / len(ranks)
            avg_m = sum(margins) / len(margins)
            eq = sum(1 for r in ranks if r == 0) / len(ranks)
            avg_gp = sum(gold_probs) / len(gold_probs)
            avg_yp = sum(greedy_probs) / len(greedy_probs)
            lines.append(
                f"- Logged rows: **{len(ranks)}** (one per sample per eval step). "
                f"**gold==greedy** rate: **{eq:.3f}**."
            )
            lines.append(
                f"- **Mean gold token rank** (0=greedy): **{avg_r:.2f}**; "
                f"mean **logit margin** (greedy−gold): **{avg_m:.4f}**."
            )
            lines.append(
                f"- Mean **prob(gold)** **{avg_gp:.4f}**, mean **prob(greedy)** **{avg_yp:.4f}** "
                f"(gap **{avg_yp - avg_gp:.4f}**)."
            )
            lines.append(
                "- *Interpretation:* rank≈1–2 with small margin often means greedy is brittle though mass is near gold; "
                "large rank means the first answer token is not yet behaviorally locked under this prompt format."
            )
            lines.extend(
                [
                    "",
                    "### Q2 answer",
                    "",
                    f"_Aggregate over {len(ranks)} rows: mean rank {avg_r:.2f}, mean greedy−gold logit margin {avg_m:.4f}, "
                    f"mean prob gap (greedy−gold) {avg_yp - avg_gp:.4f}._",
                    "",
                ]
            )
    else:
        lines.append("_No first_token_margin_over_time.csv._")
    lines.extend(["", "## Q3 — Prefix rollouts: does forcing 1/3/5 gold tokens fix generation?", ""])
    pr_csv = _manifest_path("prefix_rollout_summary.csv")
    if pr_csv is not None and pr_csv.is_file():
        by_pl: Dict[str, List[Dict[str, str]]] = {}
        with pr_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                pl = row.get("rollout_prefix_len", "?")
                by_pl.setdefault(pl, []).append(row)
        for pl in sorted(by_pl.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
            rows = by_pl[pl]
            last_row = rows[-1]
            clamp_note = ""
            try:
                nc = int(str(last_row.get("n_clamped_short_gold") or "0"))
                if nc > 0:
                    clamp_note = f", clamped_short_gold={nc}"
            except ValueError:
                pass
            lines.append(
                f"- **prefix_len={pl}** (last eval step in log): "
                f"strict_string_EM={last_row.get('raw_em_count')}/8, "
                f"diagnostic_norm_EM={last_row.get('normalized_em_count')}/8, "
                f"prefix1={last_row.get('prefix1_acc')}, "
                f"token_f1_mean={last_row.get('token_f1_mean')}, "
                f"lcs_overlap_mean={last_row.get('lcs_overlap_mean')}{clamp_note}"
            )
        lines.append(
            "- *Interpretation:* a large jump from prefix_len 0→1 implicates **first-token / early-step** instability; "
            "little gain through 5 suggests **deeper** trajectory mismatch beyond the first few tokens."
        )
        lines.extend(["", "### Q3 answer", "", "_See table above: compare normalized_EM and prefix acc across prefix_len 0 vs 1 vs 3 vs 5._", ""])
    else:
        lines.append("_No prefix_rollout_summary.csv._")
    lines.extend(["", "## Q4 — Decode ablation: does beam beat greedy?", ""])
    dab = _manifest_path("decode_ablation.csv")
    if dab is not None and dab.is_file():
        with dab.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lines.append(
                    f"- **{row.get('decode_name')}**: strict_string_EM={row.get('raw_em_count')}/8, "
                    f"trimmed={row.get('trimmed_em_count')}/8, "
                    f"diagnostic_norm={row.get('normalized_em_count')}/8, "
                    f"prefix1={row.get('prefix1_acc')}"
                )
        lines.append(
            "- *Interpretation:* if beam materially raises EM / F1, the reference string may sit in the "
            "distribution but **greedy path** is unstable. If beam barely helps, the model may not assign enough "
            "mass to the reference continuation under this decode setup."
        )
        lines.extend(["", "### Q4 answer", "", "_Compare greedy vs beam2 vs beam4 rows above on diagnostic_norm and prefix1._", ""])
    else:
        lines.append("_No decode_ablation.csv._")
    lines.extend(["", "## Q5 — Overfit ladder: when does behavior break (1/2/4/8)?", ""])
    lad = _manifest_path("overfit_ladder.csv")
    ladder_break = "unknown (missing CSV)"
    if lad is not None and lad.is_file():
        rows = list(csv.DictReader(lad.open(encoding="utf-8")))
        for row in rows:
            raw_c = row.get("raw_em_count", row.get("baseline_normalize_em_count", ""))
            norm_c = row.get("normalized_em_count", row.get("diagnostic_normalized_em_count", ""))
            base_c = row.get("baseline_eval_normalize_em_count", "")
            lines.append(
                f"- **n={row.get('ladder_n')}** ({row.get('ladder_steps')} steps): "
                f"final_train_loss={row.get('final_train_loss')}, "
                f"tf_shifted_acc={row.get('teacher_forced_shifted_token_acc_mean')}, "
                f"raw_EM={raw_c}/8, diagnostic_norm_EM={norm_c}/8"
                + (f", baseline_eval_norm_EM={base_c}/8" if base_c != "" else "")
                + f", p1/p3/p5={row.get('prefix1_acc')}/{row.get('prefix3_acc')}/{row.get('prefix5_acc')}, "
                f"F1={row.get('token_f1_mean')}, LCS={row.get('lcs_overlap_mean')}"
            )
        try:
            def _raw_em(r: Dict[str, str]) -> int:
                if r.get("raw_em_count", "") != "":
                    return int(float(r["raw_em_count"]))
                return int(float(r.get("baseline_normalize_em_count", 0)))

            n1 = next((x for x in rows if str(x.get("ladder_n")) == "1"), None)
            n8 = next((x for x in rows if str(x.get("ladder_n")) == "8"), None)
            if n1 is not None and _raw_em(n1) < 1:
                ladder_break = "raw EM fails already at n=1 (fundamental decode/alignment issue)"
            elif n8 is not None and _raw_em(n8) < 8:
                ladder_break = "n=1 may pass but n=8 still below 8/8 raw EM (interference / capacity / trajectory)"
            else:
                ladder_break = "compare n=1,2,4,8 raw_EM and tf_acc in table for first break point"
        except (ValueError, StopIteration):
            ladder_break = "see ladder rows above"
    else:
        lines.append("_No overfit_ladder.csv._")
    lines.extend(["", "### Q5 answer", "", f"_{ladder_break}_", ""])
    lines.extend(
        [
            "",
            "## Q6 — Primary bottleneck (heuristic → one label)",
            "",
            "Allowed labels: decode-path instability; first-token margin too weak; reference normalization mismatch; "
            "stop-condition / output-format mismatch; deeper sequence-level behavior mismatch not solved by token-level fitting.",
            "",
        ]
    )
    bottleneck = "deeper sequence-level behavior mismatch not solved by token-level fitting"
    ranks_for_q6 = ranks if ranks else []
    if (ft_csv is not None and ft_csv.is_file()) and (pr_csv is not None and pr_csv.is_file()):
        try:
            mean_r = sum(ranks_for_q6) / max(1, len(ranks_for_q6))
            p0 = p1 = None
            with pr_csv.open(encoding="utf-8") as f:
                pr_rows = list(csv.DictReader(f))
            if pr_rows:
                by_step_pl: Dict[Tuple[int, int], str] = {}
                for row in pr_rows:
                    try:
                        by_step_pl[(int(row["step"]), int(row["rollout_prefix_len"]))] = row.get(
                            "normalized_em_count", "0"
                        )
                    except (KeyError, ValueError):
                        continue
                max_step = max((k[0] for k in by_step_pl), default=0)
                p0 = by_step_pl.get((max_step, 0))
                p1 = by_step_pl.get((max_step, 1))
            if p0 is not None and p1 is not None:
                try:
                    if int(p1) - int(p0) >= 2:
                        bottleneck = "first-token margin too weak / decode-path instability"
                except ValueError:
                    pass
            if mean_r > 8:
                bottleneck = "first-token margin too weak"
            if dab is not None and dab.is_file():
                drows = list(csv.DictReader(dab.open(encoding="utf-8")))
                if len(drows) >= 2:
                    g = drows[0].get("normalized_em_count", 0)
                    b = drows[-1].get("normalized_em_count", 0)
                    try:
                        if int(b) - int(g) >= 2:
                            bottleneck = "decode-path instability"
                    except ValueError:
                        pass
            if last_summary_row:
                try:
                    if int(last_summary_row.get("trimmed_em_count", 0)) > int(
                        last_summary_row.get("raw_em_count", 0)
                    ) and int(last_summary_row.get("normalized_em_count", 0)) > int(
                        last_summary_row.get("trimmed_em_count", 0)
                    ):
                        bottleneck = "stop-condition / output-format mismatch"
                    if int(last_summary_row.get("normalized_em_count", 0)) > int(
                        last_summary_row.get("raw_em_count", 0)
                    ) and int(last_summary_row.get("count_whitespace_or_newline_mismatch", 0)) >= int(
                        last_summary_row.get("num_samples", 1)
                    ) // 2:
                        bottleneck = "reference normalization mismatch"
                except ValueError:
                    pass
        except Exception:
            pass
    lines.extend(
        [
            "",
            "### Q6 — Direct conclusion (single primary bottleneck)",
            "",
            f"**{bottleneck}**",
            "",
        ]
    )
    out_path = Path(results_dir) / "debug_report_sequence_behavior_diagnosis.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
