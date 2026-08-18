from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple
from core.causal_lm_metrics import (
    count_supervised_label_tokens,
    teacher_forced_token_accuracy_shifted,
)
from core.train_labels import build_supervised_labels
from core.formatting import format_for_infer


def apply_prefix_mixing_to_supervised_tokens(
    *,
    full_ids: List[int],
    labels: List[int],
    self_prefix_ids: List[int],
    prefix_k: int,
) -> Dict[str, Any]:
    """
    Replace first k supervised positions with self-fed token ids.

    Returns a dict with mixed ids/labels and bookkeeping that can be unit-tested
    independently from the HF model forward pass.
    """
    out_ids = list(full_ids)
    out_labels = list(labels)
    sup_positions = [i for i, v in enumerate(out_labels) if int(v) != -100]
    take = min(max(0, int(prefix_k)), len(sup_positions), len(self_prefix_ids))
    replaced_positions: List[int] = []
    for pos, tid in zip(sup_positions[:take], self_prefix_ids[:take]):
        out_ids[pos] = int(tid)
        out_labels[pos] = int(tid)
        replaced_positions.append(int(pos))
    return {
        "mixed_full_ids": out_ids,
        "mixed_labels": out_labels,
        "replaced_positions": replaced_positions,
        "effective_k": int(take),
    }


class BaseBackbone:
    """
    A minimal backbone interface used by the unified pipeline.

    Debug path:
      - uses a tiny, deterministic text model that can be "trained" quickly on CPU
    Future path:
      - wrap a Hugging Face causal LM (Llama 3.1 8B Instruct) while keeping the API stable
    """

    def fit_batch(self, pairs: List[Tuple[str, str]], targets: List[str], lr: float) -> Dict[str, float]:
        raise NotImplementedError

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        *,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        raise NotImplementedError

    def generate_with_ids(
        self, prompts: List[str], max_new_tokens: int = 64
    ) -> List[Dict[str, Any]]:
        """
        Debug helper for auditing prompt boundary + continuation slicing.

        Returns per-sample dicts that must include:
          - infer_prompt_token_ids
          - generated_full_ids
          - generated_continuation_ids
          - raw_generated_text
        """
        raise NotImplementedError

    def get_activations(self, prompts: List[str]) -> List[List[float]]:
        """Return per-prompt activation vectors (used by overlap/diversity regularization)."""
        raise NotImplementedError

    def score_answer_nlls(self, pairs: List[Tuple[str, str]], targets: List[str]) -> List[float]:
        """Teacher-forced mean NLL per example on the supervised answer span."""
        raise NotImplementedError


@dataclass
class DebugTextModelConfig:
    """Config for the local debug model."""

    feature_dim: int = 64
    max_memory: int = 5000


class _DebugTokenizer:
    """Tiny chat-template shim used only by the local debug model."""

    pad_token_id = 0
    eos_token_id = 1
    all_special_ids = [0, 1]

    def apply_chat_template(self, messages: List[Dict[str, str]], tokenize: bool = False, add_generation_prompt: bool = True):
        if tokenize:
            raise ValueError("_DebugTokenizer only supports tokenize=False")
        lines: List[str] = []
        for msg in messages:
            role = str(msg.get("role", "")).upper()
            content = str(msg.get("content", ""))
            lines.append(f"{role}: {content}")
        if add_generation_prompt:
            lines.append("ASSISTANT:")
        return "\n".join(lines)


class DebugTextModel(BaseBackbone):
    """
    A very lightweight model for local debug:

    - "Training" stores a capped key-value memory of (prompt -> output).
    - "Generation" returns memorized outputs when available, otherwise a simple heuristic.
    - "Activations" are hashed bag-of-words vectors (stable across runs with the same seed).

    This keeps the pipeline fully runnable without downloading large weights.
    """

    def __init__(self, cfg: DebugTextModelConfig, seed: int = 0):
        import numpy as np

        self.cfg = cfg
        self._rng = np.random.RandomState(seed)
        self._memory: Dict[str, str] = {}
        self._memory_fifo: List[str] = []
        self.tokenizer = _DebugTokenizer()

    def fit_batch(self, pairs: List[Tuple[str, str]], targets: List[str], lr: float) -> Dict[str, float]:
        # lr exists to keep API compatible with real optimizers.
        # Here, we simply store mappings.
        correct = 0
        for (prompt, _), y in zip(pairs, targets):
            pred = self._memory.get(prompt, "")
            if pred.strip() == y.strip():
                correct += 1
            self._remember(prompt, y)
        acc = correct / max(1, len(targets))
        return {"train_batch_acc": acc}

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        *,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        _ = num_beams  # debug model ignores beam width
        _ = do_sample
        outs: List[str] = []
        for p in prompts:
            if p in self._memory:
                outs.append(self._memory[p])
            else:
                outs.append(self._fallback(p))
        return outs

    def generate_with_ids(
        self, prompts: List[str], max_new_tokens: int = 64
    ) -> List[Dict[str, Any]]:
        # Debug model has no real tokenizer/ids; return placeholders.
        outs = self.generate(prompts, max_new_tokens=max_new_tokens)
        return [
            {
                "infer_prompt_token_ids": [],
                "infer_prompt_token_len": 0,
                "generated_full_ids": [],
                "generated_continuation_ids": [],
                "decoded_prompt_tail": "",
                "decoded_continuation_head": "",
                "raw_generated_text": out,
            }
            for out in outs
        ]

    def get_activations(self, prompts: List[str]) -> List[List[float]]:
        import numpy as np

        acts: List[List[float]] = []
        for p in prompts:
            v = np.zeros(self.cfg.feature_dim, dtype=np.float32)
            for tok in _simple_tokenize(p):
                idx = (hash(tok) % self.cfg.feature_dim + self.cfg.feature_dim) % self.cfg.feature_dim
                v[idx] += 1.0
            # L2 normalize
            norm = float(np.linalg.norm(v) + 1e-8)
            v = v / norm
            acts.append(v.tolist())
        return acts

    def score_answer_nlls(self, pairs: List[Tuple[str, str]], targets: List[str]) -> List[float]:
        scores: List[float] = []
        for (prompt, _), target in zip(pairs, targets):
            pred = self._memory.get(prompt, self._fallback(prompt))
            pred_tokens = _simple_tokenize(pred)
            gold_tokens = _simple_tokenize(target)
            if not gold_tokens:
                scores.append(0.0)
                continue
            overlap = len(set(pred_tokens) & set(gold_tokens))
            precision = overlap / max(1, len(set(pred_tokens)))
            recall = overlap / max(1, len(set(gold_tokens)))
            if precision + recall <= 0:
                f1 = 0.0
            else:
                f1 = 2.0 * precision * recall / (precision + recall)
            scores.append(float(max(0.0, 1.0 - f1)))
        return scores

    def _remember(self, prompt: str, output: str) -> None:
        if prompt in self._memory:
            self._memory[prompt] = output
            return
        self._memory[prompt] = output
        self._memory_fifo.append(prompt)
        if len(self._memory_fifo) > self.cfg.max_memory:
            old = self._memory_fifo.pop(0)
            self._memory.pop(old, None)

    def _fallback(self, prompt: str) -> str:
        # A small deterministic fallback: extract numbers and sum if asked, else generic.
        import re

        nums = [int(x) for x in re.findall(r"-?\d+", prompt)]
        if "和" in prompt or "sum" in prompt.lower():
            if len(nums) >= 2:
                return str(sum(nums[:2]))
        return "（debug 模型）我还不会，但我已记录该指令用于后续学习。"


def _simple_tokenize(text: str) -> List[str]:
    text = text.strip().lower()
    if not text:
        return []
    # extremely simple tokenization (works for mixed zh/en in a debug-only way)
    import re

    return [t for t in re.split(r"[^a-z0-9\u4e00-\u9fff]+", text) if t]


@dataclass
class HFCausalLMConfig:
    hf_model_name_or_path: str
    torch_dtype: str = "bfloat16"  # string in YAML
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    max_seq_len: int = 2048
    gen_max_new_tokens: int = 64
    # Deterministic greedy decoding defaults (overfit / eval debugging).
    gen_do_sample: bool = False
    gen_num_beams: int = 1
    debug_print_formatted_examples: bool = False
    debug_print_tokenized_examples: bool = False
    debug_max_tokenized_examples: int = 2
    debug_nan_guard: bool = False
    debug_nan_dump_dir: str = ""
    min_target_tokens_for_loss: int = 16
    debug_alignment_dump_dir: str = ""
    # Debug-only: token-level supervision/span audit for assistant boundary issues.
    mask_eos_token_in_labels: bool = True
    mask_all_special_tokens_in_labels: bool = True
    debug_alignment_token_audit_max_examples: int = 3
    # manual: mask by infer-prompt token length; completion_only: mask through response_template (HF/TRL-style)
    train_labeling_mode: str = "manual"
    completion_only_response_template: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # Minimal exposure-bias mitigation hook (debug-oriented, deterministic greedy only).
    enable_scheduled_sampling_prefix_mixing: bool = False
    scheduled_sampling_prefix_k: int = 0
    scheduled_sampling_apply_every_n_steps: int = 1
    scheduled_sampling_debug_log: bool = True


class HFCausalLMBackbone(BaseBackbone):
    """
    HuggingFace causal LM backbone that supports LoRA adapters (PEFT).

    This backbone is designed to work with the repo's existing train loop:
      - `fit_batch(...)` computes loss and calls `loss.backward()`
      - `LoRAWrapper.step_adapter()` is responsible for optimizer.step() + zero_grad()
    """

    def __init__(self, cfg: HFCausalLMConfig, *, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self._last_lr: float = 0.0
        self._debug_printed_formatted = False
        self._debug_printed_tokenized = False
        self._nan_batch_counter = 0
        # Token-level alignment audit (append until we hit max_examples, then write once).
        self._alignment_token_audit_rows: List[Dict[str, Any]] = []
        self._alignment_token_audit_written: bool = False
        self._fit_step_counter: int = 0

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_name_or_path, use_fast=True)
        # Decoder-only LMs generally require left padding for stable generation behavior.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Causal LM padding uses EOS as pad
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Device + dtype
        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        self.device = torch.device(device)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(cfg.torch_dtype).lower(), torch.bfloat16)

        # Model: load base causal LM (LoRA wrapper will attach PEFT model later).
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.hf_model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.train()

        # Training should not rely on KV cache
        if hasattr(self.model, "config") and getattr(self.model.config, "use_cache", None):
            self.model.config.use_cache = False

        # Seed (best-effort; full determinism is not guaranteed across kernels)
        try:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    def attach_peft_model(self, peft_model: Any) -> None:
        # Replace internal model reference with PEFT model.
        self.model = peft_model
        self.model.to(self.device)
        self.model.train()

    def fit_batch(self, pairs: List[Tuple[str, str]], targets: List[str], lr: float) -> Dict[str, float]:
        import torch

        # Persist lr for LoRAWrapper to pick up during optimizer.step()
        self._last_lr = float(lr)

        instructions = [str(ins) for (ins, _) in pairs]
        input_texts = [str(inp) for (_, inp) in pairs]
        max_len = int(self.cfg.max_seq_len)

        prompt_texts: List[str] = []
        full_texts: List[str] = []
        input_ids_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        attn_list: List[List[int]] = []
        decoded_supervised_spans: List[str] = []
        supervised_positions_per_example: List[List[int]] = []
        self._fit_step_counter += 1
        use_sched = bool(self.cfg.enable_scheduled_sampling_prefix_mixing) and int(self.cfg.scheduled_sampling_prefix_k) > 0
        apply_every = max(1, int(self.cfg.scheduled_sampling_apply_every_n_steps))
        apply_sched_this_step = use_sched and (self._fit_step_counter % apply_every == 0)
        scheduled_sampling_applied_count = 0

        for ins, inp, tgt in zip(instructions, input_texts, targets):
            enc = build_supervised_labels(
                self.tokenizer,
                ins,
                inp,
                str(tgt),
                max_len=max_len,
                min_target_tokens=int(self.cfg.min_target_tokens_for_loss),
                mask_eos_token_in_labels=bool(self.cfg.mask_eos_token_in_labels),
                mask_all_special_tokens_in_labels=bool(self.cfg.mask_all_special_tokens_in_labels),
                labeling_mode=str(self.cfg.train_labeling_mode),
                completion_only_response_template=str(self.cfg.completion_only_response_template),
            )
            if apply_sched_this_step:
                mixed = self._build_scheduled_sampling_mixed_encoding(
                    instruction=ins,
                    input_text=inp,
                    target=str(tgt),
                    base_enc=enc,
                    prefix_k=max(0, int(self.cfg.scheduled_sampling_prefix_k)),
                )
                if mixed is not None:
                    enc = mixed
                    scheduled_sampling_applied_count += 1
            prompt_texts.append(enc.prompt_text)
            full_texts.append(enc.full_text)
            full_ids = enc.full_ids
            labels = enc.labels
            attn = [1] * len(full_ids)

            input_ids_list.append(full_ids)
            labels_list.append(labels)
            attn_list.append(attn)
            sup_positions = [i for i, v in enumerate(labels) if v != -100]
            supervised_positions_per_example.append(sup_positions)
            if sup_positions:
                sup_ids = [full_ids[i] for i in sup_positions]
                decoded_supervised_spans.append(self.tokenizer.decode(sup_ids, skip_special_tokens=True))
            else:
                decoded_supervised_spans.append("")

        if self.cfg.debug_print_formatted_examples and not self._debug_printed_formatted:
            show_n = min(2, len(prompt_texts))
            print("\n[debug] Formatted training examples (before tokenization):")
            for i in range(show_n):
                print(f"[debug] example_{i}.prompt={repr(prompt_texts[i])}")
                print(f"[debug] example_{i}.target={repr(str(targets[i]))}")
            self._debug_printed_formatted = True

        def _get_alignment_helper(input_ids: List[int], labels_for_audit: List[int]) -> Dict[str, Any]:
            supervised_positions = [i for i, v in enumerate(labels_for_audit) if v != -100]
            if not supervised_positions:
                supervised_start_idx = len(labels_for_audit)
                supervised_end_idx = -1
                prompt_token_len = len(labels_for_audit)
                decoded_prompt_tail = ""
                decoded_supervised_head = ""
            else:
                supervised_start_idx = int(min(supervised_positions))
                supervised_end_idx = int(max(supervised_positions))
                prompt_token_len = supervised_start_idx
                prompt_tail_start = max(0, supervised_start_idx - 30)
                supervised_head_end = min(len(input_ids), supervised_start_idx + 30)
                decoded_prompt_tail = self.tokenizer.decode(
                    input_ids[prompt_tail_start:supervised_start_idx], skip_special_tokens=False
                )
                decoded_supervised_head = self.tokenizer.decode(
                    input_ids[supervised_start_idx:supervised_head_end], skip_special_tokens=True
                )

            return {
                "prompt_token_len": int(prompt_token_len),
                "full_token_len": int(len(input_ids)),
                "supervised_start_idx": int(supervised_start_idx),
                "supervised_end_idx": int(supervised_end_idx),
                "decoded_prompt_tail": decoded_prompt_tail,
                "decoded_supervised_head": decoded_supervised_head,
            }

        batch_size = len(input_ids_list)
        pad_id = int(self.tokenizer.pad_token_id)
        max_batch_len = max(len(x) for x in input_ids_list)
        max_batch_len = min(max_batch_len, max_len)

        input_ids = torch.full((batch_size, max_batch_len), pad_id, dtype=torch.long, device=self.device)
        labels = torch.full((batch_size, max_batch_len), -100, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_batch_len), dtype=torch.long, device=self.device)

        for i, (ids, lab, attn) in enumerate(zip(input_ids_list, labels_list, attn_list)):
            cur_len = min(len(ids), max_batch_len)
            input_ids[i, :cur_len] = torch.tensor(ids[:cur_len], dtype=torch.long, device=self.device)
            labels[i, :cur_len] = torch.tensor(lab[:cur_len], dtype=torch.long, device=self.device)
            attention_mask[i, :cur_len] = torch.tensor(attn[:cur_len], dtype=torch.long, device=self.device)

        if self.cfg.debug_print_tokenized_examples and not self._debug_printed_tokenized:
            show_n = min(int(self.cfg.debug_max_tokenized_examples), batch_size)
            print("\n[debug] Tokenized examples:")
            for i in range(show_n):
                ids = input_ids[i].detach().cpu().tolist()
                labs = labels[i].detach().cpu().tolist()
                masked = [idx for idx, v in enumerate(labs) if v == -100]
                supervised = [idx for idx, v in enumerate(labs) if v != -100]
                print(f"[debug] example_{i}.input_ids={ids}")
                print(f"[debug] example_{i}.labels={labs}")
                print(f"[debug] example_{i}.masked_label_positions(-100)={masked}")
                print(f"[debug] example_{i}.num_supervised_tokens={len(supervised)}")
                print(f"[debug] example_{i}.decoded_supervised_span={repr(decoded_supervised_spans[i])}")
            self._debug_printed_tokenized = True

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = outputs.loss
        if self.cfg.debug_nan_guard:
            has_bad_loss = bool(torch.isnan(loss).any().item() or torch.isinf(loss).any().item())
            has_bad_logits = bool(torch.isnan(outputs.logits).any().item() or torch.isinf(outputs.logits).any().item())
            if has_bad_loss or has_bad_logits:
                self._nan_batch_counter += 1
                dump_dir = self.cfg.debug_nan_dump_dir or "results/runs/nan_debug"
                d = Path(dump_dir)
                d.mkdir(parents=True, exist_ok=True)
                snap = {
                    "nan_batch_counter": self._nan_batch_counter,
                    "has_bad_loss": has_bad_loss,
                    "has_bad_logits": has_bad_logits,
                    "loss": float(loss.detach().float().item()),
                    "prompts": prompt_texts,
                    "targets": [str(t) for t in targets],
                    "input_ids": input_ids.detach().cpu().tolist(),
                    "labels": labels.detach().cpu().tolist(),
                    "attention_mask": attention_mask.detach().cpu().tolist(),
                    "num_supervised_tokens": int(labels.ne(-100).sum().item()),
                }
                (d / f"nan_batch_{self._nan_batch_counter:04d}.json").write_text(
                    json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise RuntimeError(f"NaN/Inf detected in fit_batch; dumped to {str(d)}")
        loss.backward()

        with torch.no_grad():
            logits = outputs.logits  # [B, T, V]
            # Align with HF causal LM loss: compare argmax(logits[:, :-1]) to labels[:, 1:].
            answer_token_acc, _, num_loss_tokens = teacher_forced_token_accuracy_shifted(logits, labels)
            supervised_tokens = int(count_supervised_label_tokens(labels))
            total_tokens = attention_mask.sum().item()
            if supervised_tokens <= 0:
                raise RuntimeError("num_supervised_tokens=0 after chat-template alignment; check truncation or formatting.")
            if int(attention_mask.sum().item()) <= supervised_tokens:
                print("[warn] prompt length may be collapsed; total tokens nearly equals supervised tokens.")

        # Compute quick exact-match accuracy for drift signals.
        # Note: generation is expensive, but used only for small batch sizes in early experiments.
        with torch.no_grad():
            preds = self.generate(prompt_texts, max_new_tokens=int(self.cfg.gen_max_new_tokens))
            correct = 0
            for pred, y in zip(preds, targets):
                if self._normalize(pred) == self._normalize(y):
                    correct += 1
            acc = correct / max(1, len(targets))

        if self.cfg.debug_alignment_dump_dir:
            d = Path(self.cfg.debug_alignment_dump_dir)
            d.mkdir(parents=True, exist_ok=True)
            show_n = min(2, batch_size)
            rows: List[Dict[str, Any]] = []
            for i in range(show_n):
                rows.append(
                    {
                        "formatted_train_prompt": prompt_texts[i],
                        "formatted_train_full_text": full_texts[i],
                        "input_ids": input_ids[i].detach().cpu().tolist(),
                        "labels": labels[i].detach().cpu().tolist(),
                        "supervised_token_indices": supervised_positions_per_example[i],
                        "decoded_supervised_span": decoded_supervised_spans[i],
                        "num_total_tokens": int(attention_mask[i].sum().item()),
                        "num_supervised_tokens": int(labels[i].ne(-100).sum().item()),
                        "target_text": str(targets[i]),
                    }
                )
            (d / "train_alignment_examples.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        # Token-level assistant boundary audit (debug only).
        if bool(self.cfg.debug_alignment_dump_dir) and not self._alignment_token_audit_written:
            out_dir = Path(self.cfg.debug_alignment_dump_dir).parent  # keep required path shape: .../debug/alignment_token_audit.json
            out_dir.mkdir(parents=True, exist_ok=True)
            max_rows = max(0, int(self.cfg.debug_alignment_token_audit_max_examples))

            # Collect across fit_batch calls until we have enough examples.
            for i in range(min(batch_size, max_rows - len(self._alignment_token_audit_rows))):
                helper = _get_alignment_helper(input_ids_list[i], labels_list[i])
                sup_positions = supervised_positions_per_example[i]
                decoded_supervised_span = decoded_supervised_spans[i]

                def _norm_tokens(text: str) -> List[str]:
                    t = self._normalize(text)
                    return [tok for tok in t.split() if tok]

                gold_text = str(targets[i])
                gold_tokens = _norm_tokens(gold_text)
                span_head_tokens = _norm_tokens(helper["decoded_supervised_head"])
                k = min(len(gold_tokens), len(span_head_tokens))
                span_prefix_match = bool(k > 0 and span_head_tokens[:k] == gold_tokens[:k])

                # Heuristic: if span head contains template control markers, we flag it.
                contains_control_markers = "<|" in helper["decoded_supervised_head"]
                includes_control_tokens_unexpected = bool(contains_control_markers and "<|" not in gold_text)

                self._alignment_token_audit_rows.append(
                    {
                        "formatted_train_prompt_text": prompt_texts[i],
                        "formatted_full_train_text": full_texts[i],
                        "prompt_token_ids": input_ids_list[i][: int(helper["prompt_token_len"])],
                        "full_token_ids": input_ids_list[i],
                        "supervised_token_indices": sup_positions,
                        "decoded_supervised_span": decoded_supervised_span,
                        "gold_target_text": gold_text,
                        **helper,
                        # Invariant checks (debug only; does not hard-fail by default).
                        "span_prefix_match_with_gold": span_prefix_match,
                        "includes_control_tokens_unexpected": includes_control_tokens_unexpected,
                    }
                )

            if len(self._alignment_token_audit_rows) >= max_rows > 0:
                (out_dir / "alignment_token_audit.json").write_text(
                    json.dumps(self._alignment_token_audit_rows, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                self._alignment_token_audit_written = True

        return {
            "train_batch_acc": float(acc),
            "train_loss": float(loss.detach().item()),
            "train_answer_token_acc": float(answer_token_acc),
            "num_total_tokens": int(total_tokens),
            "num_supervised_tokens": int(supervised_tokens),
            "num_loss_tokens": int(num_loss_tokens),
            "lr": float(self._last_lr),
            "scheduled_sampling_applied_examples": int(scheduled_sampling_applied_count),
        }

    def score_answer_nlls(self, pairs: List[Tuple[str, str]], targets: List[str]) -> List[float]:
        import torch
        import torch.nn.functional as F

        if len(pairs) != len(targets):
            raise ValueError("pairs and targets must have the same length")
        if not pairs:
            return []

        instructions = [str(ins) for (ins, _) in pairs]
        input_texts = [str(inp) for (_, inp) in pairs]
        max_len = int(self.cfg.max_seq_len)

        all_nlls = []
        batch_size_chunk = 4
        
        for batch_idx in range(0, len(pairs), batch_size_chunk):
            end_idx = min(batch_idx + batch_size_chunk, len(pairs))
            b_ins = instructions[batch_idx:end_idx]
            b_inp = input_texts[batch_idx:end_idx]
            b_tgt = targets[batch_idx:end_idx]

            input_ids_list: List[List[int]] = []
            labels_list: List[List[int]] = []
            attn_list: List[List[int]] = []
            for ins, inp, tgt in zip(b_ins, b_inp, b_tgt):
                enc = build_supervised_labels(
                    self.tokenizer,
                    ins,
                    inp,
                    str(tgt),
                    max_len=max_len,
                    min_target_tokens=int(self.cfg.min_target_tokens_for_loss),
                    mask_eos_token_in_labels=bool(self.cfg.mask_eos_token_in_labels),
                    mask_all_special_tokens_in_labels=bool(self.cfg.mask_all_special_tokens_in_labels),
                    labeling_mode=str(self.cfg.train_labeling_mode),
                    completion_only_response_template=str(self.cfg.completion_only_response_template),
                )
                input_ids_list.append(list(enc.full_ids))
                labels_list.append(list(enc.labels))
                attn_list.append([1] * len(enc.full_ids))

            batch_size = len(input_ids_list)
            pad_id = int(self.tokenizer.pad_token_id)
            max_batch_len = min(max(len(x) for x in input_ids_list), max_len)
            input_ids = torch.full((batch_size, max_batch_len), pad_id, dtype=torch.long, device=self.device)
            labels = torch.full((batch_size, max_batch_len), -100, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((batch_size, max_batch_len), dtype=torch.long, device=self.device)
            for i, (ids, lab, attn) in enumerate(zip(input_ids_list, labels_list, attn_list)):
                cur_len = min(len(ids), max_batch_len)
                input_ids[i, :cur_len] = torch.tensor(ids[:cur_len], dtype=torch.long, device=self.device)
                labels[i, :cur_len] = torch.tensor(lab[:cur_len], dtype=torch.long, device=self.device)
                attention_mask[i, :cur_len] = torch.tensor(attn[:cur_len], dtype=torch.long, device=self.device)

            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                token_losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(shift_labels.shape)

            nlls: List[float] = []
            for i in range(batch_size):
                mask = shift_labels[i].ne(-100)
                if bool(mask.any().item()):
                    nlls.append(float(token_losses[i][mask].mean().item()))
                else:
                    nlls.append(0.0)
                    
            all_nlls.extend(nlls)

            if was_training:
                self.model.train()
            else:
                pass
        return all_nlls

    def score_prompt_nlls(self, prompts: List[str]) -> List[float]:
        """Label-free mean NLL per token of the (already formatted) prompt text.

        Used for routing arbitration at eval time: only the input prompt is
        scored, never the gold answer, so this leaks no label information.
        """
        import torch
        import torch.nn.functional as F

        if not prompts:
            return []
        max_len = int(self.cfg.max_seq_len)
        all_nlls: List[float] = []
        for batch_idx in range(0, len(prompts), 4):
            chunk = list(prompts[batch_idx : batch_idx + 4])
            enc = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()
                shift_mask = attention_mask[:, 1:].contiguous().bool()
                token_losses = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(shift_labels.shape)
            for i in range(input_ids.size(0)):
                m = shift_mask[i]
                if bool(m.any().item()):
                    all_nlls.append(float(token_losses[i][m].mean().item()))
                else:
                    all_nlls.append(0.0)
            if was_training:
                self.model.train()
            else:
                self.model.eval()
                
        return all_nlls

    def _build_scheduled_sampling_mixed_encoding(
        self,
        *,
        instruction: str,
        input_text: str,
        target: str,
        base_enc: Any,
        prefix_k: int,
    ) -> Optional[Any]:
        """
        Minimal exposure-bias mitigation hook:
        - Greedily self-generate up to `prefix_k` answer-prefix tokens from infer prompt.
        - Replace first k supervised gold tokens with those self-fed tokens.
        - Keep supervision mask unchanged (prompt still unsupervised).

        This is intentionally conservative and debug-oriented; it is not a full
        scheduled-sampling research implementation.
        """
        import torch

        if prefix_k <= 0:
            return None
        labels = list(base_enc.labels)
        full_ids = list(base_enc.full_ids)
        sup_positions = [i for i, v in enumerate(labels) if int(v) != -100]
        if not sup_positions:
            return None
        take = min(int(prefix_k), len(sup_positions))
        prompt_text = format_for_infer(self.tokenizer, instruction, input_text, add_generation_prompt=True)
        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            # `prompt_text` already comes from apply_chat_template and includes BOS/chat markers.
            add_special_tokens=False,
            truncation=True,
            max_length=int(self.cfg.max_seq_len),
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        was_training = bool(self.model.training)
        self.model.eval()
        generated: List[int] = []
        with torch.no_grad():
            for _ in range(take):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                next_id = int(torch.argmax(out.logits[0, -1]).item())
                generated.append(next_id)
                next_t = torch.tensor([[next_id]], dtype=input_ids.dtype, device=self.device)
                input_ids = torch.cat([input_ids, next_t], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=self.device)],
                    dim=1,
                )
        if was_training:
            self.model.train()
        mixed = apply_prefix_mixing_to_supervised_tokens(
            full_ids=full_ids,
            labels=labels,
            self_prefix_ids=generated,
            prefix_k=take,
        )
        full_ids = list(mixed["mixed_full_ids"])
        labels = list(mixed["mixed_labels"])
        if bool(self.cfg.scheduled_sampling_debug_log):
            print(
                "[scheduled_sampling_prefix_mixing] "
                f"k={prefix_k} effective_k={take} "
                f"gold_head={sup_positions[:take]} mixed_ids={generated}"
            )
        # Reuse metadata from base encoding; only token ids/labels are changed.
        return type(base_enc)(
            full_ids=full_ids,
            labels=labels,
            manual_prompt_len=int(base_enc.manual_prompt_len),
            completion_supervise_start=int(base_enc.completion_supervise_start),
            prompt_text=str(base_enc.prompt_text),
            full_text=str(base_enc.full_text),
        )

    def _make_generation_config(
        self,
        max_new_tokens: int,
        *,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> Any:
        from transformers import GenerationConfig

        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
        eos_id = int(self.tokenizer.eos_token_id) if self.tokenizer.eos_token_id is not None else None
        nb = int(num_beams) if num_beams is not None else int(self.cfg.gen_num_beams)
        ds = bool(self.cfg.gen_do_sample) if do_sample is None else bool(do_sample)
        return GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=ds,
            num_beams=max(1, nb),
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
            temperature=1.0,
            top_p=1.0,
            early_stopping=nb > 1,
        )

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        *,
        num_beams: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> List[str]:
        import torch

        # Ensure eval-like generation behavior
        was_training = self.model.training
        self.model.eval()

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.max_seq_len),
            # Prompts are preformatted chat-template strings; adding special tokens again duplicates BOS.
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_cfg = self._make_generation_config(max_new_tokens, num_beams=num_beams, do_sample=do_sample)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)

        # Slice off prompt part to return only generated continuation.
        # For batched generation with padding, generated sequences are aligned to the
        # padded input width (not per-sample non-pad length), so slice by padded width.
        gen_texts: List[str] = []
        prompt_width = int(inputs["input_ids"].shape[1])
        for i in range(outputs.shape[0]):
            gen_ids = outputs[i, prompt_width:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_texts.append(text)

        if was_training:
            self.model.train()
        return gen_texts

    def generate_with_forced_answer_prefix(
        self,
        prompt: str,
        max_new_tokens: int,
        forced_answer_token_ids: List[int],
        *,
        num_beams: Optional[int] = None,
    ) -> str:
        """
        Tokenize `prompt`, append `forced_answer_token_ids`, then generate continuation.
        Returns decoded text for the assistant span starting at the first answer token position
        (includes forced prefix + model continuation), using skip_special_tokens=True.
        """
        import torch

        was_training = self.model.training
        self.model.eval()
        enc = self.tokenizer(
            prompt,
            # `prompt` already includes chat-template special tokens.
            add_special_tokens=False,
            truncation=True,
            max_length=int(self.cfg.max_seq_len),
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = int(input_ids.shape[1])
        if forced_answer_token_ids:
            ft = torch.tensor([forced_answer_token_ids], dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, ft], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, ft.shape[1], dtype=attention_mask.dtype, device=self.device)],
                dim=1,
            )
        gen_cfg = self._make_generation_config(max_new_tokens, num_beams=num_beams)
        gen_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        with torch.no_grad():
            outputs = self.model.generate(**gen_inputs, generation_config=gen_cfg)
        cont_ids = outputs[0, prompt_len:].detach().cpu().tolist()
        text = self.tokenizer.decode(cont_ids, skip_special_tokens=True)
        if was_training:
            self.model.train()
        return text

    def generate_with_ids(self, prompts: List[str], max_new_tokens: int = 64) -> List[Dict[str, Any]]:
        import torch
        from transformers import GenerationConfig

        was_training = self.model.training
        self.model.eval()

        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=int(self.cfg.max_seq_len),
            # Prompts are preformatted chat-template strings; keep tokenization identical to training.
            add_special_tokens=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_width = int(inputs["input_ids"].shape[1])

        gen_cfg = self._make_generation_config(max_new_tokens, num_beams=None)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, generation_config=gen_cfg)

        results: List[Dict[str, Any]] = []
        for i in range(outputs.shape[0]):
            full_ids = outputs[i].detach().cpu().tolist()
            continuation_ids = outputs[i, prompt_width:].detach().cpu().tolist()
            assert continuation_ids == full_ids[prompt_width:], "Continuation slicing mismatch"

            infer_prompt_token_ids = inputs["input_ids"][i].detach().cpu().tolist()
            decoded_prompt_tail = self.tokenizer.decode(
                infer_prompt_token_ids[max(0, prompt_width - 30) : prompt_width],
                skip_special_tokens=False,
            )
            decoded_continuation_head = self.tokenizer.decode(
                continuation_ids[:30],
                skip_special_tokens=False,
            )
            raw_generated_text = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)

            results.append(
                {
                    "infer_prompt_token_ids": infer_prompt_token_ids,
                    "infer_prompt_token_len": prompt_width,
                    "generated_full_ids": full_ids,
                    "generated_continuation_ids": continuation_ids,
                    "decoded_prompt_tail": decoded_prompt_tail,
                    "decoded_continuation_head": decoded_continuation_head,
                    "raw_generated_text": raw_generated_text,
                }
            )

        if was_training:
            self.model.train()
        return results

    def get_activations(self, prompts: List[str]) -> List[List[float]]:
        pooled = self.get_activations_tensor(prompts, with_grad=False)
        acts: List[List[float]] = []
        import torch

        for i in range(pooled.shape[0]):
            acts.append(pooled[i].detach().cpu().to(torch.float32).tolist())
        return acts

    def get_activations_tensor(self, prompts: List[str], *, with_grad: bool) -> "Any":
        """
        Return a pooled, L2-normalized activation tensor for cosine similarity.

        When `with_grad=True`, gradients will flow to the model parameters (needed for
        anti-overlap regularization integrated into the training step).
        """
        import torch

        was_training = self.model.training
        self.model.eval()

        all_pooled = []
        batch_size_chunk = 4

        for batch_idx in range(0, len(prompts), batch_size_chunk):
            end_idx = min(batch_idx + batch_size_chunk, len(prompts))
            b_prompts = prompts[batch_idx:end_idx]

            inputs = self.tokenizer(
                b_prompts,
                padding=True,
                truncation=True,
                max_length=int(self.cfg.max_seq_len),
                # Activations should be computed on the same prompt tokenization used by training/eval.
                add_special_tokens=False,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            if with_grad:
                outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            else:
                with torch.no_grad():
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

            hidden = outputs.hidden_states[-1]  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # [B, H]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            all_pooled.append(pooled)

        pooled = torch.cat(all_pooled, dim=0)

        # Keep training state stable
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        return pooled

    @staticmethod
    def _normalize(s: str) -> str:
        return (s or "").strip().lower()


def build_backbone(model_cfg: Dict[str, Any], *, mode: str, seed: int, debug_loading: str = "dummy") -> BaseBackbone:
    """
    Build backbone according to mode/config.

    - debug: always returns DebugTextModel unless debug_loading == "hf"
    - baseline/ours: intended to load HuggingFace model (scaffold)
    """

    if mode == "debug":
        if debug_loading == "hf":
            return build_hf_backbone(model_cfg, seed=seed)
        return DebugTextModel(DebugTextModelConfig(), seed=seed)

    return build_hf_backbone(model_cfg, seed=seed)


def build_hf_backbone(model_cfg: Dict[str, Any], *, seed: int) -> BaseBackbone:
    """
    Hugging Face backbone scaffold.

    This repo is research-ready but does not require the 8B weights at generation time.
    If you point hf_model_name_or_path to a valid local path and install dependencies,
    you can implement the real loading here with transformers.

    For now, we return the debug model with a clear error message if user truly expects HF.
    """

    hf_path = str(model_cfg.get("hf_model_name_or_path", "")).strip()
    if not hf_path:
        raise ValueError("hf_model_name_or_path is required for baseline/ours HF backbone.")

    cfg = HFCausalLMConfig(
        hf_model_name_or_path=hf_path,
        torch_dtype=str(model_cfg.get("torch_dtype", "bfloat16")),
        device=str(model_cfg.get("device", "auto")),
        max_seq_len=int(model_cfg.get("max_seq_len", 2048)),
        gen_max_new_tokens=int(model_cfg.get("gen_max_new_tokens", 64)),
        gen_do_sample=bool(model_cfg.get("gen_do_sample", False)),
        gen_num_beams=int(model_cfg.get("gen_num_beams", 1)),
        mask_eos_token_in_labels=bool(model_cfg.get("mask_eos_token_in_labels", True)),
        mask_all_special_tokens_in_labels=bool(model_cfg.get("mask_all_special_tokens_in_labels", True)),
        train_labeling_mode=str(model_cfg.get("train_labeling_mode", "manual")),
        completion_only_response_template=str(model_cfg.get("completion_only_response_template", "<|start_header_id|>assistant<|end_header_id|>\n\n")),
        enable_scheduled_sampling_prefix_mixing=bool(model_cfg.get("enable_scheduled_sampling_prefix_mixing", False)),
        scheduled_sampling_prefix_k=int(model_cfg.get("scheduled_sampling_prefix_k", 0)),
        scheduled_sampling_apply_every_n_steps=int(model_cfg.get("scheduled_sampling_apply_every_n_steps", 1)),
        scheduled_sampling_debug_log=bool(model_cfg.get("scheduled_sampling_debug_log", True)),
        debug_alignment_token_audit_max_examples=int(model_cfg.get("debug_alignment_token_audit_max_examples", 3)),
        debug_print_formatted_examples=bool(model_cfg.get("debug_print_formatted_examples", False)),
        debug_print_tokenized_examples=bool(model_cfg.get("debug_print_tokenized_examples", False)),
        debug_max_tokenized_examples=int(model_cfg.get("debug_max_tokenized_examples", 2)),
        debug_nan_guard=bool(model_cfg.get("debug_nan_guard", False)),
        debug_nan_dump_dir=str(model_cfg.get("debug_nan_dump_dir", "")),
        min_target_tokens_for_loss=int(model_cfg.get("min_target_tokens_for_loss", 16)),
        debug_alignment_dump_dir=str(model_cfg.get("debug_alignment_dump_dir", "")),
    )
    return HFCausalLMBackbone(cfg, seed=seed)

