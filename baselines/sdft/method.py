from __future__ import annotations

"""SDFT (ICML 2026): on-policy self-distillation from a demonstration-conditioned teacher.

Official TRL trainer is H200-oriented (`idanshen/Self-Distillation`; the
Continual-Intelligence URL 404s). Here: one short student rollout per example,
teacher = frozen backbone + official-style demonstration ICL prompt (stop-grad),
KL on rollout tokens. Reuses `run_opsd_step` with teacher_source='base'.
"""

from typing import Any, Dict, List

from core.data import Segment
from core.formatting import build_user_content

from baselines.basic_baselines.opsd.method import _to_pairs, _batch, run_opsd_step
from baselines.citb.common import cfg_seed, ensure_default_adapter
from baselines.lit_common import method_cfg


def format_for_sdft_teacher(tokenizer: Any, instruction: str, input_text: str, reference: str) -> str:
    """Official SDFT teacher: same user task + gold demonstration, then re-ask."""
    user_content = build_user_content(instruction, input_text)
    ref = str(reference or "").strip()
    content = (
        f"{user_content}\n\n"
        "This is an example for a response to the question:\n"
        f"{ref}\n\n"
        "Now answer the original task. Return only the requested response."
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


class SDFTMethod:
    name = "sdft"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "sdft")
        opsd_cfg = cfg.get("opsd", {}) if isinstance(cfg.get("opsd", {}), dict) else {}
        self.divergence = str(block.get("divergence", opsd_cfg.get("divergence", "kl"))).lower()
        if self.divergence not in {"kl", "jsd"}:
            self.divergence = "kl"
        self.beta = float(block.get("beta", opsd_cfg.get("beta", 0.5)))
        self.rollout_max_new_tokens = int(
            block.get("rollout_max_new_tokens", opsd_cfg.get("rollout_max_new_tokens", 32))
        )
        self.rollout_temperature = float(
            block.get("rollout_temperature", opsd_cfg.get("rollout_temperature", 1.0))
        )
        self.teacher_format_constraint = bool(
            block.get("teacher_format_constraint", opsd_cfg.get("teacher_format_constraint", True))
        )
        _ = cfg_seed(cfg)

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "sdft_divergence": self.divergence,
            "sdft_rollout_max_new_tokens": self.rollout_max_new_tokens,
        }

    def train_on_segment(
        self,
        *,
        segment: Segment,
        model: Any,
        lora: Any,
        lr: float,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        pairs, targets = _to_pairs(segment.train)
        teacher_adapter = lora.get_active_adapter_name() or "default"
        metrics: Dict[str, Any] = {"batches": 0}
        losses: List[float] = []
        grad_norms: List[float] = []
        rollout_tokens = 0
        rollout_count = 0
        for _ in range(max(1, int(epochs))):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                teacher_prompts = [
                    format_for_sdft_teacher(model.tokenizer, ins, inp, ref)
                    for (ins, inp), ref in zip(b_pairs, b_targets)
                ]
                step = run_opsd_step(
                    model=model,
                    lora=lora,
                    b_pairs=b_pairs,
                    b_targets=b_targets,
                    lr=lr,
                    divergence=self.divergence,
                    beta=self.beta,
                    rollout_temperature=self.rollout_temperature,
                    rollout_top_p=1.0,
                    rollout_max_new_tokens=self.rollout_max_new_tokens,
                    # Official DistilTrainer keeps a separate pretrained ref_model.
                    # On 4090 that is PEFT disable_adapter() (frozen backbone), not
                    # the live student LoRA — live-adapter KL collapsed to "iseise".
                    teacher_source="base",
                    teacher_adapter=teacher_adapter,
                    teacher_format_constraint=self.teacher_format_constraint,
                    teacher_prompts=teacher_prompts,
                )
                if step.get("skipped"):
                    continue
                losses.append(float(step["loss"]))
                grad_norms.append(float(step.get("grad_norm", 0.0)))
                rollout_tokens += int(step.get("rollout_token_total", 0))
                rollout_count += int(step.get("rollout_count", 0))
                metrics["batches"] += 1
        metrics["train.loss"] = sum(losses) / max(1, len(losses))
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["rollout_mean_tokens"] = rollout_tokens / max(1, rollout_count)
        metrics["active_adapter"] = teacher_adapter
        metrics["sdft_divergence"] = self.divergence
        return metrics
