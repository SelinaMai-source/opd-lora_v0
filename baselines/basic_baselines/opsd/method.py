from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.data import Example, Segment
from core.formatting import format_for_infer, format_for_teacher


class OPSDMethod:
    """
    On-Policy Self-Distillation (OPSD), following 2026-08-20-selina.md:

    - Student prompt: same as inference (`format_for_infer`), never sees the reference output.
    - Teacher prompt: student user content + privileged reference block + transition instruction
      (`format_for_teacher`), run on the frozen base via PEFT `disable_adapter()` (no extra copy).
    - Each batch: sample a rollout from the *current* student policy (do_sample, temp=1.0),
      then run a dual forward over the SAME rollout token ids and compute a per-token
      divergence D(p_T || p_S) on every rollout position (generalized JSD by default, beta=0.5;
      `divergence: kl` switches to KL(p_T || p_S)). Only the student LoRA is updated.
    """

    name = "opsd"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        opsd_cfg = cfg.get("opsd", {}) if isinstance(cfg.get("opsd", {}), dict) else {}
        self.divergence = str(opsd_cfg.get("divergence", "jsd")).strip().lower()
        if self.divergence not in {"jsd", "kl"}:
            raise ValueError(f"opsd.divergence must be 'jsd' or 'kl', got: {self.divergence}")
        self.beta = float(opsd_cfg.get("beta", 0.5))
        self.rollout_temperature = float(opsd_cfg.get("rollout_temperature", 1.0))
        self.rollout_top_p = float(opsd_cfg.get("rollout_top_p", 1.0))
        self.rollout_max_new_tokens = int(opsd_cfg.get("rollout_max_new_tokens", 0))  # 0 -> model default

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        if "default" in lora.list_adapters():
            lora.set_active_adapter("default")
        return {"active_adapter": lora.get_active_adapter_name()}

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
        import torch

        metrics: Dict[str, Any] = {"batches": 0}
        pairs, targets = _to_pairs(segment.train)

        tokenizer = model.tokenizer
        device = model.device
        max_len = int(model.cfg.max_seq_len)
        max_new = self.rollout_max_new_tokens or int(model.cfg.gen_max_new_tokens)
        pad_id = int(tokenizer.pad_token_id)
        peft_model = getattr(lora, "peft_model", None)

        batch_losses: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        rollout_token_total = 0
        rollout_count = 0

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                student_prompts = [format_for_infer(tokenizer, ins, inp) for ins, inp in b_pairs]
                teacher_prompts = [
                    format_for_teacher(tokenizer, ins, inp, ref)
                    for (ins, inp), ref in zip(b_pairs, b_targets)
                ]
                rollouts = model.sample_rollouts(
                    student_prompts,
                    max_new_tokens=max_new,
                    temperature=self.rollout_temperature,
                    top_p=self.rollout_top_p,
                )

                student_ids_list: List[List[int]] = []
                teacher_ids_list: List[List[int]] = []
                student_prompt_lens: List[int] = []
                teacher_prompt_lens: List[int] = []
                for sp, tp, ro in zip(student_prompts, teacher_prompts, rollouts):
                    if not ro:
                        continue
                    s_prompt = tokenizer(sp, add_special_tokens=False)["input_ids"]
                    t_prompt = tokenizer(tp, add_special_tokens=False)["input_ids"]
                    # Keep the rollout intact; truncate prompts from the left (tail stays,
                    # consistent with training/eval truncation) to fit max_seq_len.
                    budget = max(1, max_len - len(ro))
                    s_prompt = s_prompt[-budget:]
                    t_prompt = t_prompt[-budget:]
                    student_ids_list.append(s_prompt + ro)
                    teacher_ids_list.append(t_prompt + ro)
                    student_prompt_lens.append(len(s_prompt))
                    teacher_prompt_lens.append(len(t_prompt))
                    rollout_token_total += len(ro)
                    rollout_count += 1

                if not student_ids_list:
                    continue

                s_ids, s_mask = _pad_right(student_ids_list, pad_id=pad_id, device=device)
                t_ids, t_mask = _pad_right(teacher_ids_list, pad_id=pad_id, device=device)

                student_logits = model.forward_logits(s_ids, s_mask)  # [B, T_s, V], grad on
                with torch.no_grad():
                    if peft_model is not None and hasattr(peft_model, "disable_adapter"):
                        with peft_model.disable_adapter():
                            teacher_logits = model.forward_logits(t_ids, t_mask)
                    else:
                        teacher_logits = model.forward_logits(t_ids, t_mask)

                per_example_losses = []
                for i in range(len(student_ids_list)):
                    r_len = len(student_ids_list[i]) - student_prompt_lens[i]
                    # Logits at position P-1+j predict rollout token j (causal shift).
                    s_span = student_logits[i, student_prompt_lens[i] - 1 : student_prompt_lens[i] - 1 + r_len, :]
                    t_span = teacher_logits[i, teacher_prompt_lens[i] - 1 : teacher_prompt_lens[i] - 1 + r_len, :]
                    per_example_losses.append(
                        _token_divergence(t_span.float(), s_span.float(), kind=self.divergence, beta=self.beta)
                    )
                loss = torch.stack(per_example_losses).mean()
                if bool(torch.isnan(loss).item()) or bool(torch.isinf(loss).item()):
                    raise RuntimeError(f"OPSD loss is NaN/Inf: {float(loss.item())}")

                model._last_lr = float(lr)
                loss.backward()
                step_stats = lora.step_adapter()

                batch_losses.append(float(loss.detach().item()))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                metrics["batches"] += 1

        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["rollout_mean_tokens"] = rollout_token_total / max(1, rollout_count)
        metrics["divergence"] = self.divergence
        metrics["beta"] = float(self.beta)
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics


def _token_divergence(teacher_logits: Any, student_logits: Any, *, kind: str, beta: float) -> Any:
    """Mean per-token divergence D(p_T || p_S) over the rollout span ([R, V] logits each)."""
    import torch
    import torch.nn.functional as F

    log_t = F.log_softmax(teacher_logits, dim=-1)
    log_s = F.log_softmax(student_logits, dim=-1)
    p_t = log_t.exp()
    p_s = log_s.exp()
    if kind == "kl":
        per_token = (p_t * (log_t - log_s)).sum(dim=-1)
    else:
        b = min(max(float(beta), 1e-6), 1.0 - 1e-6)
        # m = b * p_T + (1 - b) * p_S;  JSD_b = b*KL(p_T||m) + (1-b)*KL(p_S||m)
        log_m = torch.log(b * p_t + (1.0 - b) * p_s + 1e-12)
        kl_t = (p_t * (log_t - log_m)).sum(dim=-1)
        kl_s = (p_s * (log_s - log_m)).sum(dim=-1)
        per_token = b * kl_t + (1.0 - b) * kl_s
    return per_token.mean()


def _pad_right(id_lists: List[List[int]], *, pad_id: int, device: Any) -> Tuple[Any, Any]:
    import torch

    width = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(id_lists), width), dtype=torch.long, device=device)
    for i, seq in enumerate(id_lists):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, : len(seq)] = 1
    return ids, mask


def _to_pairs(examples: List[Example]) -> Tuple[List[Tuple[str, str]], List[str]]:
    pairs: List[Tuple[str, str]] = []
    targets: List[str] = []
    for ex in examples:
        pairs.append((ex.instruction, ex.input))
        targets.append(ex.output)
    return pairs, targets


def _batch(pairs: List[Tuple[str, str]], targets: List[str], batch_size: int):
    bs = max(1, int(batch_size))
    for i in range(0, len(pairs), bs):
        yield pairs[i : i + bs], targets[i : i + bs]
