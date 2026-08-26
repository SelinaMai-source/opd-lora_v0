from __future__ import annotations

from typing import Any, Dict, List, Tuple

from baselines.basic_baselines.opsd.method import (
    _batch,
    _to_pairs,
    causal_lm_ce,
    pad_right,
    pad_right_labels,
    token_divergence,
)
from core.data import Segment
from core.formatting import format_for_teacher
from core.train_labels import build_supervised_labels, first_supervised_index


class GoldOPSDMethod:
    """
    Off-policy gold distillation: L = L_CE + λ L_KL on the same gold token sequence.

    - Student: standard SFT CE on gold continuation (same labeling as fit_batch).
    - Teacher: format_for_teacher + no_grad frozen base (disable_adapter).
    - KL(p_T || p_S) on gold tokens only; only student LoRA is updated.
    """

    name = "gold_opsd"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        opsd_cfg = cfg.get("opsd", {}) if isinstance(cfg.get("opsd", {}), dict) else {}
        gold_cfg = cfg.get("gold_opsd", {}) if isinstance(cfg.get("gold_opsd", {}), dict) else {}
        self.lambda_kl = float(gold_cfg.get("lambda_kl", 0.3))
        self.teacher_format_constraint = bool(opsd_cfg.get("teacher_format_constraint", True))

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

        pairs, targets = _to_pairs(segment.train)
        tokenizer = model.tokenizer
        device = model.device
        max_len = int(model.cfg.max_seq_len)
        pad_id = int(tokenizer.pad_token_id)
        peft_model = getattr(lora, "peft_model", None)

        metrics: Dict[str, Any] = {"batches": 0}
        ce_vals: List[float] = []
        kl_vals: List[float] = []
        loss_vals: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                s_ids_list: List[List[int]] = []
                s_labels_list: List[List[int]] = []
                gold_ids_list: List[List[int]] = []
                s_prompt_lens: List[int] = []
                t_ids_list: List[List[int]] = []
                t_prompt_lens: List[int] = []

                for (ins, inp), tgt in zip(b_pairs, b_targets):
                    enc = build_supervised_labels(
                        tokenizer,
                        ins,
                        inp,
                        str(tgt),
                        max_len=max_len,
                        min_target_tokens=int(model.cfg.min_target_tokens_for_loss),
                        mask_eos_token_in_labels=bool(model.cfg.mask_eos_token_in_labels),
                        mask_all_special_tokens_in_labels=bool(model.cfg.mask_all_special_tokens_in_labels),
                        labeling_mode=str(model.cfg.train_labeling_mode),
                        completion_only_response_template=str(model.cfg.completion_only_response_template),
                    )
                    p_len = first_supervised_index(enc.labels)
                    gold_ids = enc.full_ids[p_len:]
                    if p_len <= 0 or not gold_ids:
                        continue
                    tp = format_for_teacher(
                        tokenizer,
                        ins,
                        inp,
                        str(tgt),
                        format_constraint=self.teacher_format_constraint,
                    )
                    t_prompt = tokenizer(tp, add_special_tokens=False)["input_ids"]
                    budget = max(1, max_len - len(gold_ids))
                    t_prompt = t_prompt[-budget:]
                    s_ids_list.append(enc.full_ids)
                    s_labels_list.append(enc.labels)
                    gold_ids_list.append(gold_ids)
                    s_prompt_lens.append(p_len)
                    t_ids_list.append(t_prompt + gold_ids)
                    t_prompt_lens.append(len(t_prompt))

                if not s_ids_list:
                    continue

                s_ids, s_mask = pad_right(s_ids_list, pad_id=pad_id, device=device)
                s_labels = pad_right_labels(s_labels_list, device=device)
                t_ids, t_mask = pad_right(t_ids_list, pad_id=pad_id, device=device)

                student_logits = model.forward_logits(s_ids, s_mask)
                with torch.no_grad():
                    if peft_model is not None and hasattr(peft_model, "disable_adapter"):
                        with peft_model.disable_adapter():
                            teacher_logits = model.forward_logits(t_ids, t_mask)
                    else:
                        teacher_logits = model.forward_logits(t_ids, t_mask)

                ce = causal_lm_ce(student_logits, s_labels)
                per_kl = []
                for i in range(len(s_ids_list)):
                    r_len = len(gold_ids_list[i])
                    s_span = student_logits[i, s_prompt_lens[i] - 1 : s_prompt_lens[i] - 1 + r_len, :]
                    t_span = teacher_logits[i, t_prompt_lens[i] - 1 : t_prompt_lens[i] - 1 + r_len, :]
                    per_kl.append(token_divergence(t_span.float(), s_span.float(), kind="kl", beta=0.5))
                kl = torch.stack(per_kl).mean()
                loss = ce + float(self.lambda_kl) * kl
                if bool(torch.isnan(loss).item()) or bool(torch.isinf(loss).item()):
                    raise RuntimeError(f"gold_opsd loss is NaN/Inf: {float(loss.item())}")

                model._last_lr = float(lr)
                loss.backward()
                step_stats = lora.step_adapter()

                ce_vals.append(float(ce.detach().item()))
                kl_vals.append(float(kl.detach().item()))
                loss_vals.append(float(loss.detach().item()))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                metrics["batches"] += 1

        metrics["train.loss"] = sum(loss_vals) / max(1, len(loss_vals))
        metrics["train.ce"] = sum(ce_vals) / max(1, len(ce_vals))
        metrics["train.kl"] = sum(kl_vals) / max(1, len(kl_vals))
        metrics["lambda_kl"] = float(self.lambda_kl)
        metrics["teacher_format_constraint"] = bool(self.teacher_format_constraint)
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        return metrics
