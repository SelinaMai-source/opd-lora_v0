"""Surface-format normalization for eval, applied symmetrically to pred and gold.

Only strips wrapping / punctuation / chatty prefixes / whitespace. Does not
rewrite answer semantics (no paraphrase, no dropping content words, no
conversational 'Sure,' which is often gold dialogue).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Tuple

# Longest-first. Leading meta-answer wrappers only — not dialogue "sure,".
ANSWER_PREFIXES: Tuple[str, ...] = (
    "the correct answer is: ",
    "the correct answer is ",
    "the correct option is: ",
    "the correct option is ",
    "the correct label is: ",
    "the correct label is ",
    "the output is: ",
    "the output is ",
    "the answer is: ",
    "the answer is ",
    "my answer is: ",
    "my answer is ",
    "here's the answer: ",
    "here's the answer ",
    "here is the answer: ",
    "here is the answer ",
    "final answer: ",
    "final answer ",
    "prediction: ",
    "answer: ",
    "label: ",
    "output: ",
)

_QUOTE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('"', '"'),
    ("'", "'"),
    ("`", "`"),
    ("“", "”"),
    ("‘", "’"),
)

_AFTER_STEP_PERIOD = re.compile(r"(?i)\b(after\s+step\s*\d+)\s*\.+\s*$")
_AFTER_STEP_ANY = re.compile(r"(?i)\bafter\s+step\s*(\d+)\b")
_TRAILING_PUNCT = re.compile(r"[\s\.。;；,，:：!！?？]+$")
_NUMBERED_CHOICE = re.compile(r"^(?:\(?[A-Da-d]\)|[1-9]\)|\([1-9]\))\s+")
_EXPL_HEAD = re.compile(
    r"(?is)^(because|since|explanation\b|note:|this is because|here'?s why|the reason)\b"
)
_MARKDOWN_WRAP = (
    re.compile(r"^\*\*(.+)\*\*$", re.DOTALL),
    re.compile(r"^__(.+)__$", re.DOTALL),
    re.compile(r"^`(.+)`$", re.DOTALL),
)


def strip_outer_quotes(text: str) -> str:
    out = str(text or "").strip()
    changed = True
    while changed and len(out) >= 2:
        changed = False
        for left, right in _QUOTE_PAIRS:
            if out.startswith(left) and out.endswith(right) and len(out) >= len(left) + len(right):
                inner = out[len(left) : len(out) - len(right)].strip()
                if inner != out:
                    out = inner
                    changed = True
                    break
    return out


def strip_answer_prefixes(text: str) -> str:
    out = str(text or "").strip()
    for _ in range(4):
        lowered = out.lower()
        matched = False
        for pref in ANSWER_PREFIXES:
            if lowered.startswith(pref):
                out = out[len(pref) :].strip()
                matched = True
                break
        if not matched:
            break
    return out


def strip_trailing_punct(text: str) -> str:
    return _TRAILING_PUNCT.sub("", str(text or "").strip())


def normalize_after_step_period(text: str) -> str:
    """'After step 5.' / 'After step 5.' → 'After step 5' (period-only)."""
    return _AFTER_STEP_PERIOD.sub(r"\1", str(text or "").strip())


def drop_extra_explanation_lines(text: str) -> str:
    """Keep the first line when it is a short structured answer plus ramble.

    Multi-line golds (dialogue / generation) are left intact unless the first
    line is After-step / a short label and the rest looks like an explanation.
    """
    raw = str(text or "")
    if "\n" not in raw:
        return raw.strip()
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return raw.strip()
    first = lines[0]
    rest = " ".join(lines[1:])
    if _AFTER_STEP_ANY.search(first):
        return first
    if len(first.split()) <= 8 and _EXPL_HEAD.match(rest):
        return first
    parts = re.split(r"\n\s*\n", raw.strip(), maxsplit=1)
    if len(parts) == 2 and len(parts[0].split()) <= 8 and _EXPL_HEAD.match(parts[1].strip()):
        return parts[0].strip()
    return raw.strip()


def unwrap_label_wrapping(text: str) -> str:
    """Unwrap markdown / numbered-choice / quoted-list wrappers around a label."""
    out = str(text or "").strip()
    for pat in _MARKDOWN_WRAP:
        m = pat.fullmatch(out)
        if m:
            out = m.group(1).strip()
            break
    out = _NUMBERED_CHOICE.sub("", out).strip()
    # ['i', 'n', 't'] → i, n, t  (and [1, 2] → 1, 2 on both sides)
    if len(out) >= 2 and out[0] == "[" and out[-1] == "]":
        inner = out[1:-1]
        if "[" not in inner and "]" not in inner:
            parts = [p.strip().strip("'\"") for p in inner.split(",")]
            if parts and all(parts):
                out = ", ".join(parts)
    return out


def normalize_output_format(text: str) -> str:
    """Rule-based surface format normalize; safe to apply to pred and gold."""
    from core.normalize_answer import drop_eos_tokens

    out = drop_eos_tokens(str(text or ""))
    out = unicodedata.normalize("NFKC", out)
    out = out.replace("\u00a0", " ").replace("\ufeff", "")
    out = out.strip()
    out = drop_extra_explanation_lines(out)
    for _ in range(3):
        nxt = strip_answer_prefixes(out)
        nxt = unwrap_label_wrapping(nxt)
        nxt = strip_outer_quotes(nxt)
        nxt = nxt.strip()
        if nxt == out:
            break
        out = nxt
    out = normalize_after_step_period(out)
    out = strip_trailing_punct(out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
