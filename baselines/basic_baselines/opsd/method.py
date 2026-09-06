from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

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
        self.teacher_format_constraint = bool(opsd_cfg.get("teacher_format_constraint", False))

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
        metrics: Dict[str, Any] = {"batches": 0}
        pairs, targets = _to_pairs(segment.train)

        batch_losses: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        rollout_token_total = 0
        rollout_count = 0

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                step = run_opsd_step(
                    model=model,
                    lora=lora,
                    b_pairs=b_pairs,
                    b_targets=b_targets,
                    lr=lr,
                    divergence=self.divergence,
                    beta=self.beta,
                    rollout_temperature=self.rollout_temperature,
                    rollout_top_p=self.rollout_top_p,
                    rollout_max_new_tokens=self.rollout_max_new_tokens,
                    teacher_source="base",
                    teacher_format_constraint=self.teacher_format_constraint,
                )
                if step.get("skipped"):
                    continue
                batch_losses.append(float(step["loss"]))
                grad_norms.append(float(step["grad_norm"]))
                lr_values.append(float(step["lr"]))
                rollout_token_total += int(step["rollout_token_total"])
                rollout_count += int(step["rollout_count"])
                metrics["batches"] += 1

        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["rollout_mean_tokens"] = rollout_token_total / max(1, rollout_count)
        metrics["divergence"] = self.divergence
        metrics["beta"] = float(self.beta)
        metrics["teacher_format_constraint"] = bool(self.teacher_format_constraint)
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics


def run_opsd_step(
    *,
    model: Any,
    lora: Any,
    b_pairs: List[Tuple[str, str]],
    b_targets: List[str],
    lr: float,
    divergence: str,
    beta: float,
    rollout_temperature: float,
    rollout_top_p: float,
    rollout_max_new_tokens: int,
    teacher_source: str = "base",
    teacher_adapter: str = "teacher",
    teacher_format_constraint: bool = False,
    rollouts: Optional[List[List[int]]] = None,
    do_step: bool = True,
    teacher_prototypes: Optional[List[Optional[Sequence[Mapping[str, Any]]]]] = None,
) -> Dict[str, Any]:
    """
    One OPSD optimizer step (rollout + dual forward + divergence + backward + step).

    teacher_source:
      - "base": frozen backbone via PEFT disable_adapter()
      - "adapter": frozen LoRA named `teacher_adapter` (must already exist)

    rollouts: optional precomputed continuation token ids (one list per pair). When
      set, sampling is skipped so a caller can pick teacher golds from the same
      on-policy sample.
    do_step: if False, skip backward/optimizer and return `loss_tensor` for the
      caller to combine (e.g. CE + λ OPSD) before a single step.
    """
    import torch

    if teacher_source not in {"base", "adapter"}:
        raise ValueError(f"teacher_source must be 'base' or 'adapter', got: {teacher_source}")

    tokenizer = model.tokenizer
    device = model.device
    max_len = int(model.cfg.max_seq_len)
    max_new = rollout_max_new_tokens or int(model.cfg.gen_max_new_tokens)
    pad_id = int(tokenizer.pad_token_id)
    peft_model = getattr(lora, "peft_model", None)

    student_prompts = [format_for_infer(tokenizer, ins, inp) for ins, inp in b_pairs]
    teacher_prompts = []
    for i, ((ins, inp), ref) in enumerate(zip(b_pairs, b_targets)):
        protos = None
        if teacher_prototypes is not None and i < len(teacher_prototypes):
            protos = teacher_prototypes[i]
        teacher_prompts.append(
            format_for_teacher(
                tokenizer,
                ins,
                inp,
                ref,
                format_constraint=teacher_format_constraint,
                prototypes=protos,
            )
        )
    if rollouts is None:
        rollouts = model.sample_rollouts(
            student_prompts,
            max_new_tokens=max_new,
            temperature=rollout_temperature,
            top_p=rollout_top_p,
        )

    student_ids_list: List[List[int]] = []
    teacher_ids_list: List[List[int]] = []
    student_prompt_lens: List[int] = []
    teacher_prompt_lens: List[int] = []
    rollout_token_total = 0
    rollout_count = 0
    for sp, tp, ro in zip(student_prompts, teacher_prompts, rollouts):
        if not ro:
            continue
        s_prompt = tokenizer(sp, add_special_tokens=False)["input_ids"]
        t_prompt = tokenizer(tp, add_special_tokens=False)["input_ids"]
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
        return {"skipped": True}

    s_ids, s_mask = pad_right(student_ids_list, pad_id=pad_id, device=device)
    t_ids, t_mask = pad_right(teacher_ids_list, pad_id=pad_id, device=device)

    student_logits = model.forward_logits(s_ids, s_mask)
    teacher_logits = _teacher_forward_logits(
        model,
        lora,
        t_ids,
        t_mask,
        teacher_source=teacher_source,
        teacher_adapter=teacher_adapter,
        peft_model=peft_model,
    )

    per_example_losses = []
    for i in range(len(student_ids_list)):
        r_len = len(student_ids_list[i]) - student_prompt_lens[i]
        s_span = student_logits[i, student_prompt_lens[i] - 1 : student_prompt_lens[i] - 1 + r_len, :]
        t_span = teacher_logits[i, teacher_prompt_lens[i] - 1 : teacher_prompt_lens[i] - 1 + r_len, :]
        per_example_losses.append(
            token_divergence(t_span.float(), s_span.float(), kind=divergence, beta=beta)
        )
    loss = torch.stack(per_example_losses).mean()
    if bool(torch.isnan(loss).item()) or bool(torch.isinf(loss).item()):
        raise RuntimeError(f"OPSD loss is NaN/Inf: {float(loss.item())}")

    if not do_step:
        return {
            "skipped": False,
            "loss": float(loss.detach().item()),
            "loss_tensor": loss,
            "rollout_token_total": int(rollout_token_total),
            "rollout_count": int(rollout_count),
        }

    model._last_lr = float(lr)
    loss.backward()
    step_stats = lora.step_adapter()
    return {
        "skipped": False,
        "loss": float(loss.detach().item()),
        "grad_norm": float(step_stats.get("grad_norm", 0.0)),
        "lr": float(step_stats.get("lr", lr)),
        "rollout_token_total": int(rollout_token_total),
        "rollout_count": int(rollout_count),
    }


def _teacher_forward_logits(
    model: Any,
    lora: Any,
    t_ids: Any,
    t_mask: Any,
    *,
    teacher_source: str,
    teacher_adapter: str,
    peft_model: Any,
) -> Any:
    import torch

    with torch.no_grad():
        if teacher_source == "adapter":
            # Switch PEFT active adapter only. Do not call lora.set_active_adapter():
            # that toggles requires_grad on student LoRA and can drop student grads.
            if peft_model is not None and hasattr(peft_model, "set_adapter"):
                peft_model.set_adapter(teacher_adapter)
                try:
                    return model.forward_logits(t_ids, t_mask)
                finally:
                    peft_model.set_adapter(lora.get_active_adapter_name() or "default")
            prev = lora.get_active_adapter_name()
            lora.set_active_adapter(teacher_adapter)
            try:
                return model.forward_logits(t_ids, t_mask)
            finally:
                lora.set_active_adapter(prev)
        if peft_model is not None and hasattr(peft_model, "disable_adapter"):
            with peft_model.disable_adapter():
                return model.forward_logits(t_ids, t_mask)
        return model.forward_logits(t_ids, t_mask)


def token_divergence(teacher_logits: Any, student_logits: Any, *, kind: str, beta: float) -> Any:
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
        log_m = torch.log(b * p_t + (1.0 - b) * p_s + 1e-12)
        kl_t = (p_t * (log_t - log_m)).sum(dim=-1)
        kl_s = (p_s * (log_s - log_m)).sum(dim=-1)
        per_token = b * kl_t + (1.0 - b) * kl_s
    return per_token.mean()


def pad_right(id_lists: List[List[int]], *, pad_id: int, device: Any) -> Tuple[Any, Any]:
    import torch

    width = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(id_lists), width), dtype=torch.long, device=device)
    for i, seq in enumerate(id_lists):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[i, : len(seq)] = 1
    return ids, mask


def pad_right_labels(label_lists: List[List[int]], *, device: Any) -> Any:
    import torch

    width = max(len(x) for x in label_lists)
    labels = torch.full((len(label_lists), width), -100, dtype=torch.long, device=device)
    for i, lab in enumerate(label_lists):
        labels[i, : len(lab)] = torch.tensor(lab, dtype=torch.long, device=device)
    return labels


def causal_lm_ce(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
    )


def token_mean_logprob(logits: Any, labels: Any, *, chunk_size: int = 16) -> Any:
    """Length-normalized mean token log-prob on supervised positions (labels != -100).

    Uses fused cross_entropy per short chunk so a float32 [B, T, vocab]
    logsumexp table is never materialized (OOM on CNN/DM + Llama-8B vocab:
    chunk=128 * V * 4B = 64MiB extra).
    """
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    tgt = labels[:, 1:]
    mask = tgt != -100
    seq_len = int(shift_logits.size(1))
    cs = max(1, int(chunk_size))
    pieces = []
    for start in range(0, seq_len, cs):
        end = min(start + cs, seq_len)
        chunk = shift_logits[:, start:end, :]
        tgt_chunk = tgt[:, start:end]
        nll = F.cross_entropy(
            chunk.reshape(-1, chunk.size(-1)),
            tgt_chunk.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(tgt_chunk.shape)
        pieces.append(-nll)
        del chunk, nll
    token_lp = torch.cat(pieces, dim=-1) if pieces else shift_logits.new_zeros(shift_logits.size(0), 0)
    token_lp = token_lp * mask.to(dtype=token_lp.dtype)
    denom = mask.sum(dim=-1).clamp(min=1).to(dtype=token_lp.dtype)
    return (token_lp.sum(dim=-1) / denom).to(dtype=logits.dtype)


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
