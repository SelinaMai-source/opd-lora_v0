from __future__ import annotations

"""KPIG (ACL 2025): replay lowest instruction-key-part IG + masked JSD.

No official repo. Paper-faithful: mask longest tokens as key-part, IG =
CE(full) - CE(masked) on a frozen snapshot, keep K=10 lowest-IG examples,
train CE + λ JSD(student || stopgrad snapshot) on masked replay.
"""

import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from core.data import Segment
from core.train_labels import build_supervised_labels

from baselines.citb.common import (
    batch_pairs,
    cfg_seed,
    empty_cuda_cache,
    ensure_default_adapter,
    summarize_sft_metrics,
    to_pairs,
    zero_lora_grad,
)
from baselines.lit_common import mask_key_parts, method_cfg, replay_k, snapshot_adapter

# Char-length heuristic for CNN/DailyMail-style inputs: drop to micro-batch 1.
_LONG_SEQ_CHARS = 2500
# Token-chunk width for masked JSD softmax (never keep two full-vocab logit cubes).
_JSD_TOKEN_CHUNK = 16


class KPIGMethod:
    name = "kpig"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "kpig")
        self.k = replay_k(cfg, "kpig", default=10)
        self.lambda_jsd = float(block.get("lambda_jsd", 0.3))
        self.mask_frac = float(block.get("mask_frac", 0.3))
        self._rng = random.Random(cfg_seed(cfg))
        self._memory_pairs: List[Tuple[str, str]] = []
        self._memory_targets: List[str] = []
        self._snapshot: Dict[str, torch.Tensor] = {}
        self._grad_ckpt = False

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = segment
        self._enable_gradient_checkpointing(model, lora)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "kpig_k": self.k,
            "kpig_memory": len(self._memory_pairs),
            "kpig_has_snapshot": bool(self._snapshot),
            "kpig_grad_ckpt": self._grad_ckpt,
        }

    def _enable_gradient_checkpointing(self, model: Any, lora: Any) -> None:
        if self._grad_ckpt:
            return
        peft_model = getattr(lora, "peft_model", None) or getattr(model, "model", None)
        if peft_model is None:
            return
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
        candidates = [peft_model, getattr(model, "model", None)]
        if hasattr(peft_model, "get_base_model"):
            try:
                candidates.append(peft_model.get_base_model())
            except Exception:
                candidates.append(getattr(peft_model, "base_model", None))
        enabled = False
        for mod in candidates:
            if mod is None:
                continue
            cfg = getattr(mod, "config", None)
            if cfg is not None and hasattr(cfg, "use_cache"):
                cfg.use_cache = False
            if not hasattr(mod, "gradient_checkpointing_enable"):
                continue
            try:
                mod.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                enabled = True
            except TypeError:
                mod.gradient_checkpointing_enable()
                enabled = True
        self._grad_ckpt = bool(enabled)

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
        self._enable_gradient_checkpointing(model, lora)
        pairs, targets = to_pairs(segment.train)
        mix_pairs = list(pairs) + list(self._memory_pairs)
        mix_targets = list(targets) + list(self._memory_targets)
        batch_accs: List[float] = []
        batch_losses: List[float] = []
        answer_token_accs: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        jsd_vals: List[float] = []
        total_tokens = 0
        supervised_tokens = 0
        batches = 0
        model._last_lr = float(lr)
        use_jsd = bool(self._snapshot) and bool(self._memory_pairs)

        for _ in range(max(1, int(epochs))):
            idxs = list(range(len(mix_pairs)))
            if len(idxs) > 1:
                self._rng.shuffle(idxs)
            order_pairs = [mix_pairs[i] for i in idxs]
            order_targets = [mix_targets[i] for i in idxs]
            for b_pairs, b_targets in batch_pairs(order_pairs, order_targets, batch_size):
                zero_lora_grad(lora)
                long_batch = self._is_long_batch(b_pairs, b_targets)
                micro_bs = 1 if long_batch else max(1, len(b_pairs))
                n_items = max(1, len(b_pairs))
                ce_loss_sum = 0.0
                acc_sum = 0.0
                tokacc_sum = 0.0
                jsd_sum = 0.0
                tok_sum = 0
                sup_sum = 0
                n_micro = 0
                for mb_pairs, mb_targets in batch_pairs(b_pairs, b_targets, micro_bs):
                    scale = float(len(mb_pairs)) / float(n_items)
                    ce_loss, stats = model.compute_supervised_loss(mb_pairs, mb_targets)
                    (ce_loss * scale).backward()
                    jsd = ce_loss.new_zeros(())
                    if use_jsd:
                        jsd = self._masked_jsd(model, lora, mb_pairs, mb_targets)
                        jsd_term = self.lambda_jsd * jsd * scale
                        if jsd_term.requires_grad:
                            jsd_term.backward()
                    ce_loss_sum += float(stats.get("train_loss", 0.0)) * len(mb_pairs)
                    acc_sum += float(stats.get("train_batch_acc", 0.0)) * len(mb_pairs)
                    tokacc_sum += float(stats.get("train_answer_token_acc", 0.0)) * len(mb_pairs)
                    jsd_sum += float(jsd.detach().item()) * len(mb_pairs)
                    tok_sum += int(stats.get("num_total_tokens", 0))
                    sup_sum += int(stats.get("num_supervised_tokens", 0))
                    n_micro += 1
                    del ce_loss, jsd
                    if long_batch:
                        empty_cuda_cache()
                step_stats = lora.step_adapter()
                denom = float(n_items)
                batch_accs.append(acc_sum / denom)
                batch_losses.append(ce_loss_sum / denom)
                answer_token_accs.append(tokacc_sum / denom)
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                jsd_vals.append(jsd_sum / denom)
                total_tokens += tok_sum
                supervised_tokens += sup_sum
                batches += 1
                if long_batch:
                    empty_cuda_cache()

        empty_cuda_cache()
        keep_p, keep_t = self._select_low_ig(model, pairs, targets)
        self._memory_pairs.extend(keep_p)
        self._memory_targets.extend(keep_t)
        self._snapshot = snapshot_adapter(lora, lora.get_active_adapter_name())
        empty_cuda_cache()
        return summarize_sft_metrics(
            batches=batches,
            batch_accs=batch_accs,
            batch_losses=batch_losses,
            answer_token_accs=answer_token_accs,
            grad_norms=grad_norms,
            lr_values=lr_values,
            total_tokens=total_tokens,
            supervised_tokens=supervised_tokens,
            extra={
                "kpig_jsd_mean": sum(jsd_vals) / max(1, len(jsd_vals)),
                "kpig_added": len(keep_p),
                "kpig_memory": len(self._memory_pairs),
                "active_adapter": lora.get_active_adapter_name(),
            },
        )

    @staticmethod
    def _is_long_batch(pairs: List[Tuple[str, str]], targets: List[str]) -> bool:
        for (ins, inp), tgt in zip(pairs, targets):
            if len(ins) + len(inp) + len(str(tgt)) >= _LONG_SEQ_CHARS:
                return True
        return False

    def _select_low_ig(
        self,
        model: Any,
        pairs: List[Tuple[str, str]],
        targets: List[str],
    ) -> Tuple[List[Tuple[str, str]], List[str]]:
        if self.k <= 0 or not pairs:
            return [], []
        scores: List[Tuple[float, int]] = []
        was_training = bool(model.model.training)
        model.model.eval()
        with torch.no_grad():
            for i, (pair, tgt) in enumerate(zip(pairs, targets)):
                full_loss, _ = model.compute_supervised_loss([pair], [tgt])
                masked = (mask_key_parts(pair[0], self.mask_frac), pair[1])
                mask_loss, _ = model.compute_supervised_loss([masked], [tgt])
                ig = float(full_loss.detach().item()) - float(mask_loss.detach().item())
                scores.append((ig, i))
                if (i + 1) % 8 == 0:
                    empty_cuda_cache()
        if was_training:
            model.model.train()
        scores.sort(key=lambda x: x[0])
        take = [i for _, i in scores[: min(self.k, len(scores))]]
        return [pairs[i] for i in take], [targets[i] for i in take]

    def _encode_masked(
        self,
        model: Any,
        pairs: List[Tuple[str, str]],
        targets: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenizer = model.tokenizer
        max_len = int(model.cfg.max_seq_len)
        pad_id = int(tokenizer.pad_token_id)
        ids_list: List[List[int]] = []
        lab_list: List[List[int]] = []
        for (ins, inp), tgt in zip(pairs, targets):
            enc = build_supervised_labels(
                tokenizer,
                mask_key_parts(ins, self.mask_frac),
                inp,
                str(tgt),
                max_len=max_len,
                min_target_tokens=int(model.cfg.min_target_tokens_for_loss),
                mask_eos_token_in_labels=bool(model.cfg.mask_eos_token_in_labels),
                mask_all_special_tokens_in_labels=bool(model.cfg.mask_all_special_tokens_in_labels),
                labeling_mode=str(model.cfg.train_labeling_mode),
                completion_only_response_template=str(model.cfg.completion_only_response_template),
            )
            ids_list.append(list(enc.full_ids))
            lab_list.append(list(enc.labels))
        width = min(max(len(x) for x in ids_list), max_len)
        input_ids = torch.full((len(ids_list), width), pad_id, dtype=torch.long, device=model.device)
        labels = torch.full((len(ids_list), width), -100, dtype=torch.long, device=model.device)
        attn = torch.zeros((len(ids_list), width), dtype=torch.long, device=model.device)
        for i, (ids, lab) in enumerate(zip(ids_list, lab_list)):
            n = min(len(ids), width)
            input_ids[i, :n] = torch.tensor(ids[:n], dtype=torch.long, device=model.device)
            labels[i, :n] = torch.tensor(lab[:n], dtype=torch.long, device=model.device)
            attn[i, :n] = 1
        return input_ids, labels, attn

    def _masked_jsd(
        self,
        model: Any,
        lora: Any,
        pairs: List[Tuple[str, str]],
        targets: List[str],
    ) -> torch.Tensor:
        # Slice ultra-long samples instead of skipping the segment.
        if len(pairs) > 1 and self._is_long_batch(pairs, targets):
            parts = [self._masked_jsd(model, lora, [p], [t]) for p, t in zip(pairs, targets)]
            stacked = torch.stack(parts)
            return stacked.mean()

        input_ids, labels, attn = self._encode_masked(model, pairs, targets)
        mask = labels[:, 1:] != -100
        if not bool(mask.any()):
            return input_ids.new_zeros((), dtype=torch.float32)
        long_seq = int(input_ids.shape[1]) >= 768

        # Teacher first (no grad); keep only supervised-token logits, then free.
        teacher_logits = self._snapshot_forward(model, lora, input_ids, attn)
        t_sel = teacher_logits[:, :-1, :][mask].to(dtype=torch.bfloat16).contiguous()
        del teacher_logits
        if long_seq:
            empty_cuda_cache()

        student_logits = model.forward_logits(input_ids, attn)
        s_sel = student_logits[:, :-1, :][mask].to(dtype=torch.bfloat16)
        del student_logits
        if long_seq:
            empty_cuda_cache()

        return self._jsd_chunks(s_sel, t_sel)

    @staticmethod
    def _jsd_chunks(s_sel: torch.Tensor, t_sel: torch.Tensor) -> torch.Tensor:
        n = int(s_sel.shape[0])
        if n <= 0:
            return s_sel.new_zeros((), dtype=torch.float32)
        parts: List[torch.Tensor] = []
        for i in range(0, n, _JSD_TOKEN_CHUNK):
            s = s_sel[i : i + _JSD_TOKEN_CHUNK]
            t = t_sel[i : i + _JSD_TOKEN_CHUNK]
            log_s = F.log_softmax(s, dim=-1)
            log_t = F.log_softmax(t, dim=-1)
            p_s = log_s.exp()
            p_t = log_t.exp()
            log_m = torch.log(0.5 * (p_s + p_t) + 1e-12)
            jsd = 0.5 * (p_s * (log_s - log_m)).sum(-1) + 0.5 * (p_t * (log_t - log_m)).sum(-1)
            parts.append(jsd)
            del s, t, log_s, log_t, p_s, p_t, log_m
        return torch.cat(parts).float().mean()

    def _snapshot_forward(self, model: Any, lora: Any, input_ids: Any, attn: Any) -> torch.Tensor:
        peft_model = getattr(lora, "peft_model", None)
        if peft_model is None or not self._snapshot:
            with torch.no_grad():
                return model.forward_logits(input_ids, attn)
        named = dict(peft_model.named_parameters())
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for n, snap in self._snapshot.items():
                if n not in named:
                    continue
                backup[n] = named[n].data.detach().clone()
                named[n].data.copy_(snap.to(device=named[n].device, dtype=named[n].dtype))
            logits = model.forward_logits(input_ids, attn)
            for n, data in backup.items():
                named[n].data.copy_(data)
        return logits.detach()
