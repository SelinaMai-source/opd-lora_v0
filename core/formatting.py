from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence


def build_user_content(
    instruction: str,
    input_text: str,
    *,
    include_instruction: bool = True,
) -> str:
    inp = str(input_text or "").strip()
    if not include_instruction:
        return f"Input:\n{inp}" if inp else ""

    ins = str(instruction or "").strip()
    constraint = "You must answer as concisely as possible without any explanations or conversational fillers."
    if constraint not in ins:
        ins = f"{ins}\n\n{constraint}"

    if inp:
        return f"{ins}\n\nInput:\n{inp}"
    return ins


def build_chat_messages(
    instruction: str,
    input_text: str,
    target: str | None = None,
    *,
    include_instruction: bool = True,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": build_user_content(
                instruction, input_text, include_instruction=include_instruction
            ),
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": str(target)})
    return messages


def format_for_infer(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    *,
    add_generation_prompt: bool = True,
    include_instruction: bool = True,
) -> str:
    messages = build_chat_messages(
        instruction, input_text, target=None, include_instruction=include_instruction
    )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )


TEACHER_TRANSITION_INSTRUCTION = (
    "The reference output above is one valid response rather than the only acceptable wording. "
    "Use it to understand the intended meaning, tone, and degree of brevity.\n\n"
    "Now answer the original task. Do not quote, discuss, or mention the reference output. "
    "Return only the requested response."
)

TEACHER_FORMAT_CONSTRAINT = (
    "Preserve its key content, answer format, wording style, and approximate length."
)


def _format_teacher_prototypes(prototypes: Optional[Sequence[Mapping[str, Any]]]) -> str:
    """1–2 old-task exemplars for the teacher only; empty input yields no extra text."""
    if not prototypes:
        return ""
    lines = [
        "<old_task_prototypes>",
        "The following are related past-task exemplars, not the current answer. "
        "Use them only as format and style hints.",
    ]
    for i, proto in enumerate(prototypes, start=1):
        if not isinstance(proto, Mapping):
            continue
        score_type = str(proto.get("task_score_type", "") or "").strip()
        gold_len = proto.get("mean_gold_len", proto.get("gold_len", ""))
        ins = str(proto.get("instruction", "") or "").strip()
        inp = str(proto.get("input", proto.get("input_text", "")) or "").strip()
        out = str(proto.get("output", proto.get("target", "")) or "").strip()
        header = f"[{i}]"
        extras = []
        if score_type:
            extras.append(f"task_score_type={score_type}")
        if gold_len != "":
            extras.append(f"gold_len≈{gold_len}")
        if extras:
            header = f"{header} {'; '.join(extras)}"
        lines.append(header)
        if ins:
            lines.append(f"Instruction: {ins}")
        if inp:
            lines.append(f"Input: {inp}")
        if out:
            lines.append(f"Output: {out}")
    lines.append("</old_task_prototypes>")
    if len(lines) <= 3:
        return ""
    return "\n".join(lines)


def format_for_teacher(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    reference: str,
    *,
    format_constraint: bool = False,
    prototypes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> str:
    """OPSD teacher prompt: student prompt + privileged reference + optional old-task prototypes."""
    user_content = build_user_content(instruction, input_text)
    ref = str(reference or "").strip()
    transition = TEACHER_TRANSITION_INSTRUCTION
    if format_constraint:
        transition = f"{transition}\n{TEACHER_FORMAT_CONSTRAINT}"
    proto_block = _format_teacher_prototypes(prototypes)
    parts = [
        user_content,
        f"<privileged_reference_output>\n{ref}\n</privileged_reference_output>",
    ]
    if proto_block:
        parts.append(proto_block)
    parts.append(transition)
    user_content = "\n\n".join(parts)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_for_train(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    target: str,
    *,
    include_instruction: bool = True,
) -> Dict[str, str]:
    prompt_text = format_for_infer(
        tokenizer, instruction, input_text, include_instruction=include_instruction
    )
    full_text = tokenizer.apply_chat_template(
        build_chat_messages(
            instruction, input_text, target=str(target), include_instruction=include_instruction
        ),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"prompt_text": prompt_text, "full_text": full_text}
