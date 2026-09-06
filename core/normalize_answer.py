"""Unified answer normalization for eval scoring (train + LFPT5 bridge)."""
from __future__ import annotations

import re
import string
import unicodedata
from typing import Any, Dict, Optional

# Imported lazily inside normalize_for_eval to avoid a circular import:
# output_format_normalize.py uses drop_eos_tokens from this module.

_EOS_TOKENS = (
    "<|eot_id|>",
    "<s>",
    "</s>",
    "<pad>",
    "<unk>",
    "[PAD]",
    "[EOS]",
    "[BOS]",
    "[pad]",
    "[eos]",
    "[bos]",
)


def drop_eos_tokens(text: str) -> str:
    out = str(text or "")
    for tok in _EOS_TOKENS:
        out = out.replace(tok, "")
    return out


def drop_prompt_copy(
    text: str,
    prompt: str,
    *,
    drop_copy: bool = True,
    drop_eos_from_copy: bool = True,
) -> str:
    """Remove task-prefix echo from generations before whitespace normalization."""
    out = str(text or "")
    if not out or not drop_copy or not prompt:
        return out
    if prompt in out:
        copied = out
        if drop_eos_from_copy:
            copied = drop_eos_tokens(copied)
        out = copied.replace(prompt, "", 1)
    return drop_eos_tokens(out) if drop_copy else out


def normalize_for_eval(
    text: str,
    *,
    prompt: str = "",
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Normalize prediction/gold for EM/F1.

    Contract keys (eval_normalization):
      strip, drop_copy, collapse_whitespace, remove_punctuation, unicode_nfkc, drop_eos,
      format_normalize (default True; surface wrapping/punct/prefixes, pred+gold symmetric).

    Legacy aliases still honored: strip_whitespace, lowercase, remove_special_tokens,
    remove_prompt_prefix, keep_text_after_output_marker, truncate_*.

    evaluate.py scores via this function; keep format rules here rather than
    duplicating them in the scoring loop.
    """
    c = dict(cfg or {})
    out = str(text or "")

    if bool(c.get("keep_text_after_output_marker", True)):
        marker = "输出："
        if marker in out:
            out = out.split(marker)[-1]

    if bool(c.get("truncate_at_first_blankline", False)):
        out = out.split("\n\n", 1)[0]
    elif bool(c.get("truncate_at_first_newline", False)):
        out = out.split("\n", 1)[0]
    if bool(c.get("truncate_at_first_sentence_end", False)):
        m = re.search(r"[.!?]", out)
        if m is not None:
            out = out[: m.end()]

    if bool(c.get("drop_copy", False)):
        out = drop_prompt_copy(
            out,
            prompt,
            drop_copy=True,
            drop_eos_from_copy=bool(c.get("drop_eos_from_copy", True)),
        )
    elif bool(c.get("drop_eos", False)):
        out = drop_eos_tokens(out)

    if bool(c.get("remove_prompt_prefix", False)) and prompt and out.startswith(prompt):
        out = out[len(prompt) :]

    if bool(c.get("remove_special_tokens", True)):
        out = drop_eos_tokens(out)

    if bool(c.get("unicode_nfkc", False)):
        out = unicodedata.normalize("NFKC", out)

    if bool(c.get("remove_punctuation", False)):
        out = out.translate(str.maketrans("", "", string.punctuation))

    if bool(c.get("collapse_whitespace", False)):
        out = re.sub(r"\s+", " ", out)

    do_strip = bool(c.get("strip", c.get("strip_whitespace", True)))
    if do_strip:
        out = out.strip()

    if bool(c.get("lowercase", True)):
        out = out.lower()

    if bool(c.get("collapse_whitespace", False)):
        out = re.sub(r"\s+", " ", out).strip()

    # Default on for subsequent eval. Surface format only (see output_format_normalize).
    if bool(c.get("format_normalize", True)):
        from core.output_format_normalize import normalize_output_format

        out = normalize_output_format(out)

    return out


def basic_answer_normalize(text: str) -> str:
    """Lightweight normalize for audit/heuristics (no cfg)."""
    out = drop_eos_tokens(str(text or "")).strip().lower()
    out = re.sub(r"\s+", " ", out)
    return out.strip(" \t\r\n\"'`。，,.!?;:()[]{}")
