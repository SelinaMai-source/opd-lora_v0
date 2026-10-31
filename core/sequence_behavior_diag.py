"""
Diagnostic-only text normalization and failure tagging for overfit-8 sequence behavior.
Does NOT change main baseline EM (_normalize in evaluate.py).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

# Llama 3.x style stop strings (trim continuation at earliest occurrence).
_STOP_MARKERS = (
    "<|eot_id|>",
    "<|end_of_text|>",
    "</s>",
    "<|im_end|>",
)


def diagnostic_strip_assistant_boilerplate(text: str) -> str:
    """Remove obvious assistant wrapper prefixes when they are pure wrappers."""
    t = text or ""
    t = t.lstrip()
    patterns = (
        r"(?i)^assistant\s*:\s*",
        r"(?i)^assistant\s+",
        r"^<\|assistant\|>\s*",
        r"^assistant\n+",
    )
    changed = True
    while changed and t:
        changed = False
        for pat in patterns:
            m = re.match(pat, t)
            if m:
                t = t[m.end() :].lstrip()
                changed = True
    return t


def diagnostic_trim_at_stops(text: str) -> str:
    """Cut at earliest known generation stop marker (diagnostic only)."""
    t = text or ""
    earliest = None
    for m in _STOP_MARKERS:
        idx = t.find(m)
        if idx >= 0 and (earliest is None or idx < earliest):
            earliest = idx
    if earliest is not None:
        t = t[:earliest]
    return t.rstrip()


def diagnostic_collapse_whitespace(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def diagnostic_normalize(text: str) -> str:
    """
    Deterministic diagnostic normalization (NOT used for main raw EM):
    strip, strip assistant boilerplate, collapse spaces, normalize blank lines,
    NFKC, lowercase for stable comparison.
    """
    t = diagnostic_strip_assistant_boilerplate(text or "")
    t = diagnostic_trim_at_stops(t)
    t = diagnostic_collapse_whitespace(t)
    try:
        t = unicodedata.normalize("NFKC", t)
    except Exception:
        pass
    return t.lower()


def _strip_punct_tokens(s: str) -> str:
    """Remove common punctuation for punctuation-only mismatch test."""
    out: List[str] = []
    for ch in s or "":
        if ch.isalnum() or ch.isspace() or ("\u4e00" <= ch <= "\u9fff"):
            out.append(ch)
    return "".join(out)


def _tokenize_words(s: str) -> List[str]:
    return [t for t in (s or "").split() if t]


def first_mismatch_char(a: str, b: str) -> int:
    """0-based index of first differing character, or -1 if equal."""
    ca, cb = a or "", b or ""
    n = min(len(ca), len(cb))
    for i in range(n):
        if ca[i] != cb[i]:
            return i
    if len(ca) != len(cb):
        return n
    return -1


def first_mismatch_token(pred_tokens: List[str], gold_tokens: List[str]) -> int:
    n = min(len(pred_tokens), len(gold_tokens))
    for i in range(n):
        if pred_tokens[i] != gold_tokens[i]:
            return i
    if len(pred_tokens) != len(gold_tokens):
        return n
    return -1


def assign_failure_tags(
    *,
    pred_raw: str,
    gold_raw: str,
    pred_trim: str,
    gold_trim: str,
    pred_norm: str,
    gold_norm: str,
    exact_raw: bool,
    exact_trim: bool,
    exact_norm: bool,
    main_norm_pred: str,
    main_norm_gold: str,
    token_f1: float,
    lcs_overlap: float,
) -> List[str]:
    tags: List[str] = []
    pr, gr = pred_raw or "", gold_raw or ""
    pt = (pred_trim or "").strip()
    near_empty = len(pt) < 2 or len(_tokenize_words(pt)) == 0
    if near_empty:
        tags.append("empty_or_near_empty_output")
        return tags

    if exact_raw:
        return []

    pred_w = _tokenize_words(main_norm_pred)
    gold_w = _tokenize_words(main_norm_gold)

    if gold_w and pred_w:
        if pred_w[0] != gold_w[0]:
            tags.append("first_token_wrong")
        elif pred_w[:5] != gold_w[:5] and not exact_norm:
            tags.append("prefix_drift_1to5")

    if not exact_norm and token_f1 >= 0.92 and lcs_overlap >= 0.85:
        tags.append("semantic_match_surface_mismatch")

    # Extra preamble: gold appears as substring of pred after stripping boilerplate
    ptn = pred_norm or ""
    gn = gold_norm or ""
    if gn and len(gn) >= 8 and gn in ptn and ptn != gn and not exact_trim:
        # pred is longer and contains full gold as substring
        if len(ptn) > len(gn) + 5:
            tags.append("extra_preamble")

    # Over-generation / stop: trimmed much shorter than raw pred
    raw_pr = (pred_raw or "").strip()
    if len(raw_pr) > len(pt) + 15 or (diagnostic_trim_at_stops(raw_pr) != raw_pr.rstrip() and len(pt) + 10 < len(raw_pr)):
        if not exact_trim and not exact_norm:
            tags.append("wrong_stop_or_overgenerate")

    if not exact_raw and exact_norm:
        tags.append("whitespace_or_newline_mismatch")

    if (
        not exact_norm
        and _strip_punct_tokens(pred_norm) == _strip_punct_tokens(gold_norm)
        and pred_norm != gold_norm
    ):
        tags.append("punctuation_only_mismatch")

    if not exact_norm and "empty_or_near_empty_output" not in tags:
        surface_tags = {
            "first_token_wrong",
            "prefix_drift_1to5",
            "semantic_match_surface_mismatch",
            "punctuation_only_mismatch",
            "whitespace_or_newline_mismatch",
            "extra_preamble",
            "wrong_stop_or_overgenerate",
        }
        if not surface_tags.intersection(tags):
            tags.append("content_wrong")

    # Dedup preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_failure_record(
    *,
    sample_id: int,
    prompt_text: str,
    gold_text: str,
    pred_text_raw: str,
    main_norm_pred: str,
    main_norm_gold: str,
) -> Dict[str, Any]:
    from core.evaluate import _lcs_overlap, _token_f1

    gold_text = gold_text or ""
    pred_text_raw = pred_text_raw or ""

    pred_text_trimmed = diagnostic_trim_at_stops(pred_text_raw).strip()
    gold_trim_ref = diagnostic_trim_at_stops(gold_text).strip()

    pred_text_normalized = diagnostic_normalize(pred_text_raw)
    gold_text_normalized = diagnostic_normalize(gold_text)

    exact_match_raw = pred_text_raw == gold_text
    exact_match_trimmed = pred_text_trimmed == gold_trim_ref
    exact_match_normalized = pred_text_normalized == gold_text_normalized

    # Main-eval normalization (baseline EM path) for token metrics in record
    norm_pred_main = main_norm_pred
    norm_gold_main = main_norm_gold
    token_f1 = float(_token_f1(norm_pred_main, norm_gold_main))
    lcs_ov = float(_lcs_overlap(norm_pred_main, norm_gold_main))

    fc = first_mismatch_char(pred_text_raw, gold_text)
    pred_w = _tokenize_words(norm_pred_main)
    gold_w = _tokenize_words(norm_gold_main)
    ft = first_mismatch_token(pred_w, gold_w)

    tags = assign_failure_tags(
        pred_raw=pred_text_raw,
        gold_raw=gold_text,
        pred_trim=pred_text_trimmed,
        gold_trim=gold_trim_ref,
        pred_norm=pred_text_normalized,
        gold_norm=gold_text_normalized,
        exact_raw=exact_match_raw,
        exact_trim=exact_match_trimmed,
        exact_norm=exact_match_normalized,
        main_norm_pred=norm_pred_main,
        main_norm_gold=norm_gold_main,
        token_f1=token_f1,
        lcs_overlap=lcs_ov,
    )

    return {
        "sample_id": int(sample_id),
        "prompt_text": prompt_text,
        "gold_text": gold_text,
        "pred_text_raw": pred_text_raw,
        "pred_text_trimmed": pred_text_trimmed,
        "pred_text_normalized": pred_text_normalized,
        "gold_text_normalized": gold_text_normalized,
        "exact_match_raw": bool(exact_match_raw),
        "exact_match_trimmed": bool(exact_match_trimmed),
        "exact_match_normalized": bool(exact_match_normalized),
        "failure_tags": tags,
        "first_error_position_char": int(fc),
        "first_error_position_token": int(ft),
        "gold_len_chars": len(gold_text),
        "pred_len_chars": len(pred_text_raw),
        "gold_len_tokens": len(gold_w),
        "pred_len_tokens": len(pred_w),
        "token_f1_main_norm": token_f1,
        "lcs_overlap_main_norm": lcs_ov,
    }


def summarize_failure_records(records: List[Dict[str, Any]], *, step: int) -> Dict[str, Any]:
    n = len(records)
    tag_keys = (
        "first_token_wrong",
        "prefix_drift_1to5",
        "semantic_match_surface_mismatch",
        "extra_preamble",
        "wrong_stop_or_overgenerate",
        "whitespace_or_newline_mismatch",
        "punctuation_only_mismatch",
        "content_wrong",
        "empty_or_near_empty_output",
    )
    raw_em = sum(1 for r in records if r.get("exact_match_raw"))
    trim_em = sum(1 for r in records if r.get("exact_match_trimmed"))
    norm_em = sum(1 for r in records if r.get("exact_match_normalized"))
    counts = {f"count_{k}": 0 for k in tag_keys}
    for r in records:
        for t in r.get("failure_tags") or []:
            k = f"count_{t}"
            if k in counts:
                counts[k] += 1
    return {
        "step": int(step),
        "num_samples": int(n),
        "raw_em_count": int(raw_em),
        "trimmed_em_count": int(trim_em),
        "normalized_em_count": int(norm_em),
        **counts,
    }


def failure_summary_row(summary: Dict[str, Any]) -> Dict[str, Any]:
    """CSV row with exact column names from spec."""
    return {
        "step": summary["step"],
        "num_samples": summary["num_samples"],
        "raw_em_count": summary["raw_em_count"],
        "trimmed_em_count": summary["trimmed_em_count"],
        "normalized_em_count": summary["normalized_em_count"],
        "count_first_token_wrong": summary["count_first_token_wrong"],
        "count_prefix_drift_1to5": summary["count_prefix_drift_1to5"],
        "count_semantic_match_surface_mismatch": summary["count_semantic_match_surface_mismatch"],
        "count_extra_preamble": summary["count_extra_preamble"],
        "count_wrong_stop_or_overgenerate": summary["count_wrong_stop_or_overgenerate"],
        "count_whitespace_or_newline_mismatch": summary["count_whitespace_or_newline_mismatch"],
        "count_punctuation_only_mismatch": summary["count_punctuation_only_mismatch"],
        "count_content_wrong": summary["count_content_wrong"],
        "count_empty_or_near_empty_output": summary["count_empty_or_near_empty_output"],
    }
