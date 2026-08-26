from __future__ import annotations

from typing import Any, Dict, List


def build_user_content(instruction: str, input_text: str) -> str:
    ins = str(instruction or "").strip()
    inp = str(input_text or "").strip()
    
    constraint = "You must answer as concisely as possible without any explanations or conversational fillers."
    if constraint not in ins:
        ins = f"{ins}\n\n{constraint}"

    if inp:
        return f"{ins}\n\nInput:\n{inp}"
    return ins


def build_chat_messages(instruction: str, input_text: str, target: str | None = None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "user", "content": build_user_content(instruction, input_text)}]
    if target is not None:
        messages.append({"role": "assistant", "content": str(target)})
    return messages


def format_for_infer(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    messages = build_chat_messages(instruction, input_text, target=None)
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


def format_for_teacher(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    reference: str,
    *,
    format_constraint: bool = False,
) -> str:
    """OPSD teacher prompt: student prompt + privileged reference + transition instruction."""
    user_content = build_user_content(instruction, input_text)
    ref = str(reference or "").strip()
    transition = TEACHER_TRANSITION_INSTRUCTION
    if format_constraint:
        transition = f"{transition}\n{TEACHER_FORMAT_CONSTRAINT}"
    user_content = (
        f"{user_content}\n\n"
        f"<privileged_reference_output>\n{ref}\n</privileged_reference_output>\n\n"
        f"{transition}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def format_for_train(tokenizer: Any, instruction: str, input_text: str, target: str) -> Dict[str, str]:
    prompt_text = format_for_infer(tokenizer, instruction, input_text)
    full_text = tokenizer.apply_chat_template(
        build_chat_messages(instruction, input_text, target=str(target)),
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"prompt_text": prompt_text, "full_text": full_text}
