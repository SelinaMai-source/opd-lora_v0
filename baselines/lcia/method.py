from __future__ import annotations

"""
LCIA: Large Continual Instruction Assistant (ICML 2025, arXiv:2410.10868).

Two paper pieces, text-only on this repo's LoRA bank:

1. Instruction grouping (CoIN `instruct/LoRASelect.py`):
   TF-IDF on the segment's unique instructions, cosine vs codebook groups.
   Reuse the matched LoRA if max similarity >= epsilon, else spawn a new adapter.
   Eval routes each prompt to the most similar group.

2. Dynamic EMA on LoRA params (CoIN `CoIN/peft/tuners/lora.py`):
   After each optimizer step, layer-wise L1 approximation of the EMA mixing
   weight, then θ*_t = β θ*_{t-1} + (1-β) θ_t. At segment end the EMA
   weights overwrite the adapter (saved checkpoint).
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from core.data import Example, Segment

# CoIN LoRASelect.py uses 0.90 for natural-language "Instructions" templates.
_DEFAULT_SIM_THRESHOLD = 0.90
# Paper / CoIN: when β (or the derived new-weight α) leaves (0, 1), use 0.99.
_DEFAULT_FALLBACK_BETA = 0.99


@dataclass
class _InstructionGroup:
    group_idx: int
    adapter_name: str
    instructions: List[str] = field(default_factory=list)


class LCIAMethod:
    """Continual LoRA with TF-IDF instruction grouping and dynamic EMA."""

    name = "lcia"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        method_cfg = _method_cfg(cfg, "lcia")
        self.adapter_prefix = str(method_cfg.get("adapter_prefix", "lcia_g"))
        self.similarity_threshold = float(method_cfg.get("similarity_threshold", _DEFAULT_SIM_THRESHOLD))
        self.fallback_beta = float(method_cfg.get("fallback_beta", _DEFAULT_FALLBACK_BETA))
        self.freeze_inactive_adapters = bool(method_cfg.get("freeze_inactive_adapters", True))
        self.apply_ema_at_segment_end = bool(method_cfg.get("apply_ema_at_segment_end", True))
        self.tfidf_stop_words = str(method_cfg.get("tfidf_stop_words", "english"))
        # Multi-adapter PEFT + AdamW states of *all* groups OOMed on CNN/DM (seg 27).
        # Keep optimizer / EMA / prev-grad on the active group only; park the rest on CPU.
        self.optimizer_active_only = bool(method_cfg.get("optimizer_active_only", True))
        self.cpu_offload_inactive_adapters = bool(method_cfg.get("cpu_offload_inactive_adapters", True))
        self.cpu_offload_ema = bool(method_cfg.get("cpu_offload_ema", True))

        self._groups: List[_InstructionGroup] = []
        self._active_group_idx: int = -1
        self._lora: Any = None
        self._ema: Dict[str, Dict[str, Any]] = {}
        self._prev_grads: Dict[str, Dict[str, Any]] = {}
        self._eval_wrapped_models: set[int] = set()
        self._last_match_sim: float = 0.0
        self._last_reused: bool = False
        self._gpu_resident_adapter: str = ""

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        self._lora = lora
        self._install_eval_routing(model, lora)

        instructions = _unique_instructions(segment.train) or _unique_instructions(segment.eval)
        if not instructions:
            fallback = str(getattr(segment, "segment_name", "") or f"segment_{segment.segment_id}")
            instructions = [fallback]

        adapter_name, reused, match_sim = self._select_or_create_adapter(lora, instructions)
        self._last_reused = bool(reused)
        self._last_match_sim = float(match_sim)

        if self.freeze_inactive_adapters:
            self._freeze_inactive(lora, adapter_name)
        lora.set_active_adapter(adapter_name)
        if hasattr(lora, "unfreeze_adapter"):
            lora.unfreeze_adapter(adapter_name)
        self._prepare_active_adapter(lora, adapter_name)

        return {
            "active_adapter": lora.get_active_adapter_name(),
            "num_groups": len(self._groups),
            "lcia_reused_adapter": bool(reused),
            "lcia_match_sim": float(match_sim),
            "lcia_similarity_threshold": self.similarity_threshold,
            "lcia_group_idx": int(self._active_group_idx),
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
        self._lora = lora
        if self._active_group_idx < 0:
            self.on_segment_start(segment=segment, model=model, lora=lora)

        adapter_name = lora.get_active_adapter_name()
        pairs, targets = _to_pairs(segment.train)
        metrics: Dict[str, Any] = {"batches": 0, "mean_batch_acc": 0.0}
        batch_accs: List[float] = []
        batch_losses: List[float] = []
        answer_token_accs: List[float] = []
        grad_norms: List[float] = []
        lr_values: List[float] = []
        betas: List[float] = []
        total_tokens = 0
        supervised_tokens = 0
        ema_updates = 0
        ema_fallbacks = 0

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                out = model.fit_batch(b_pairs, b_targets, lr=lr)
                grad_snap = self._snapshot_grads(lora, adapter_name)
                step_stats = lora.step_adapter()
                ema_stats = self._ema_update_after_step(
                    lora=lora,
                    adapter_name=adapter_name,
                    grad_snap=grad_snap,
                )
                ema_updates += int(ema_stats.get("updated", 0))
                ema_fallbacks += int(ema_stats.get("fallbacks", 0))
                if ema_stats.get("beta_mean") is not None:
                    betas.append(float(ema_stats["beta_mean"]))
                batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                batch_losses.append(float(out.get("train_loss", 0.0)))
                answer_token_accs.append(float(out.get("train_answer_token_acc", 0.0)))
                grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
                lr_values.append(float(step_stats.get("lr", lr)))
                total_tokens += int(out.get("num_total_tokens", 0))
                supervised_tokens += int(out.get("num_supervised_tokens", 0))
                metrics["batches"] += 1

        if self.apply_ema_at_segment_end:
            self._copy_ema_into_adapter(lora, adapter_name)
        self._release_gpu_train_state(adapter_name)
        _empty_cuda_cache()

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["train.loss"] = sum(batch_losses) / max(1, len(batch_losses))
        metrics["train.answer_token_acc"] = sum(answer_token_accs) / max(1, len(answer_token_accs))
        metrics["num_total_tokens"] = int(total_tokens)
        metrics["num_supervised_tokens"] = int(supervised_tokens)
        metrics["grad_norm"] = sum(grad_norms) / max(1, len(grad_norms))
        metrics["lr"] = sum(lr_values) / max(1, len(lr_values))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["num_groups"] = len(self._groups)
        metrics["lcia_reused_adapter"] = bool(self._last_reused)
        metrics["lcia_match_sim"] = float(self._last_match_sim)
        metrics["lcia_ema_updates"] = int(ema_updates)
        metrics["lcia_ema_fallbacks"] = int(ema_fallbacks)
        metrics["lcia_beta_mean"] = sum(betas) / max(1, len(betas)) if betas else self.fallback_beta
        metrics["lcia_applied_ema_at_segment_end"] = bool(self.apply_ema_at_segment_end)
        return metrics

    def get_inference_adapter(self) -> str:
        if 0 <= self._active_group_idx < len(self._groups):
            return self._groups[self._active_group_idx].adapter_name
        return "default"

    def route_instruction(self, instruction: str) -> str:
        """Eval / debug helper: pick the codebook group with highest TF-IDF cosine."""
        return self._route_text(str(instruction or ""))

    def _select_or_create_adapter(
        self, lora: Any, instructions: List[str]
    ) -> Tuple[str, bool, float]:
        match_idx, match_sim = self._best_group(instructions)
        reused = match_idx >= 0 and match_sim >= self.similarity_threshold
        if reused:
            group = self._groups[match_idx]
            self._active_group_idx = match_idx
            for text in instructions:
                if text not in group.instructions:
                    group.instructions.append(text)
            return group.adapter_name, True, float(match_sim)

        group_idx = len(self._groups)
        adapter_name = self._adapter_name_for_index(group_idx)
        if adapter_name not in lora.list_adapters():
            lora.create_adapter(adapter_name)
        group = _InstructionGroup(
            group_idx=group_idx,
            adapter_name=adapter_name,
            instructions=list(instructions),
        )
        self._groups.append(group)
        self._active_group_idx = group_idx
        return adapter_name, False, float(match_sim)

    def _adapter_name_for_index(self, group_idx: int) -> str:
        if group_idx == 0:
            return "default"
        return f"{self.adapter_prefix}{group_idx}"

    def _best_group(self, instructions: Sequence[str]) -> Tuple[int, float]:
        if not self._groups:
            return -1, 0.0
        best_idx = 0
        best_sim = -1.0
        for i, group in enumerate(self._groups):
            sim = _mean_pairwise_tfidf_cosine(
                list(instructions),
                group.instructions,
                stop_words=self.tfidf_stop_words,
            )
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        return best_idx, float(max(0.0, best_sim))

    def _route_text(self, text: str) -> str:
        if not self._groups:
            return self.get_inference_adapter()
        stripped = str(text or "").strip()
        if stripped:
            for group in self._groups:
                for inst in group.instructions:
                    if inst and inst in stripped:
                        return group.adapter_name
        idx, _sim = self._best_group([stripped] if stripped else [""])
        if idx < 0:
            return self.get_inference_adapter()
        return self._groups[idx].adapter_name

    def _freeze_inactive(self, lora: Any, active: str) -> None:
        if not hasattr(lora, "freeze_adapter"):
            return
        for group in self._groups:
            name = group.adapter_name
            if name == active:
                continue
            if name in lora.list_adapters():
                lora.freeze_adapter(name)
        if "default" in lora.list_adapters() and active != "default":
            lora.freeze_adapter("default")

    def _prepare_active_adapter(self, lora: Any, adapter_name: str) -> None:
        self._ensure_adapter_device(lora, adapter_name)
        if self.cpu_offload_inactive_adapters:
            adapters = set(lora.list_adapters()) if hasattr(lora, "list_adapters") else set()
            for name in adapters:
                if name != adapter_name:
                    _move_adapter_params(lora, name, torch.device("cpu"))
        if self.optimizer_active_only:
            self._rebuild_active_optimizer(lora, adapter_name)
        self._gpu_resident_adapter = adapter_name
        _empty_cuda_cache()

    def _rebuild_active_optimizer(self, lora: Any, adapter_name: str) -> None:
        if not hasattr(lora, "_optimizer"):
            return
        params = [p for _, p in _iter_adapter_params(lora, adapter_name) if p.requires_grad]
        if not params:
            return
        lora._optimizer = torch.optim.AdamW(params, lr=1e-4)

    def _ensure_adapter_device(self, lora: Any, adapter_name: str) -> None:
        device = _lora_compute_device(lora)
        if device is None:
            return
        _move_adapter_params(lora, adapter_name, device)
        self._gpu_resident_adapter = adapter_name

    def _release_gpu_train_state(self, adapter_name: str) -> None:
        prev = self._prev_grads.pop(adapter_name, None)
        del prev
        ema_bank = self._ema.get(adapter_name) or {}
        if self.cpu_offload_ema:
            for key, tensor in list(ema_bank.items()):
                if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                    ema_bank[key] = tensor.detach().to("cpu")

    def _install_eval_routing(self, model: Any, lora: Any) -> None:
        """
        evaluate_stream only per-example-routes when router_only is selected.
        LCIA therefore wraps generate / generate_with_ids so seen-segment eval
        switches to the TF-IDF-matched adapter without touching evaluate.py.
        """
        model_id = id(model)
        if model_id in self._eval_wrapped_models:
            return
        orig_generate = model.generate
        orig_generate_with_ids = getattr(model, "generate_with_ids", None)

        def _routed_generate(prompts: List[str], max_new_tokens: int = 64, **kwargs: Any) -> List[str]:
            return self._generate_routed(lora, orig_generate, prompts, max_new_tokens, **kwargs)

        model.generate = _routed_generate  # type: ignore[method-assign]
        if orig_generate_with_ids is not None:

            def _routed_generate_with_ids(prompts: List[str], max_new_tokens: int = 64, **kwargs: Any):
                return self._generate_routed(
                    lora, orig_generate_with_ids, prompts, max_new_tokens, **kwargs
                )

            model.generate_with_ids = _routed_generate_with_ids  # type: ignore[method-assign]
        self._eval_wrapped_models.add(model_id)

    def _generate_routed(
        self,
        lora: Any,
        orig_fn: Callable[..., Any],
        prompts: List[str],
        max_new_tokens: int,
        **kwargs: Any,
    ) -> Any:
        if not prompts:
            return orig_fn(prompts, max_new_tokens=max_new_tokens, **kwargs)
        adapters = [self._route_text(p) for p in prompts]
        unique = list(dict.fromkeys(adapters))
        if len(unique) == 1:
            self._safe_set_adapter(lora, unique[0])
            return orig_fn(prompts, max_new_tokens=max_new_tokens, **kwargs)

        # Mixed groups in one batch: generate one prompt at a time.
        merged: List[Any] = []
        for prompt, adapter in zip(prompts, adapters):
            self._safe_set_adapter(lora, adapter)
            part = orig_fn([prompt], max_new_tokens=max_new_tokens, **kwargs)
            if isinstance(part, list):
                merged.extend(part)
            else:
                merged.append(part)
        return merged

    def _safe_set_adapter(self, lora: Any, name: str) -> None:
        adapters = set(lora.list_adapters()) if hasattr(lora, "list_adapters") else set()
        target = name if name in adapters else ("default" if "default" in adapters else "")
        if not target:
            return
        if self.cpu_offload_inactive_adapters:
            prev = self._gpu_resident_adapter
            if prev and prev != target and prev in adapters:
                _move_adapter_params(lora, prev, torch.device("cpu"))
            self._ensure_adapter_device(lora, target)
        lora.set_active_adapter(target)

    def _snapshot_grads(self, lora: Any, adapter_name: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, param in _iter_adapter_params(lora, adapter_name):
            if param.grad is None:
                continue
            out[name] = param.grad.detach().float().cpu()
        return out

    def _ema_update_after_step(
        self,
        *,
        lora: Any,
        adapter_name: str,
        grad_snap: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        CoIN `LoraModel.ema_update`, layer-wise L1 form of paper Eqs. (1)+(14):

            a = ||θ_t - θ*_{t-1}||_1
            b = ||L'(θ_t)||_1
            c = ||L'(θ_t) - L'(θ_{t-1})||_1
            α = (1+b) / (a * c) * 2     # new-param weight; clamp to 0.01 if not in (0,1)
            θ* ← (1-α) θ* + α θ
        """
        params = list(_iter_adapter_params(lora, adapter_name))
        if not params:
            return {"updated": 0, "fallbacks": 0, "beta_mean": None}

        ema_bank = self._ema.setdefault(adapter_name, {})
        prev_grads = self._prev_grads.setdefault(adapter_name, {})
        layer_groups: Dict[str, List[Tuple[str, Any]]] = {}
        for name, param in params:
            layer_groups.setdefault(_layer_key(name, adapter_name), []).append((name, param))

        betas: List[float] = []
        fallbacks = 0
        updated = 0
        fallback_alpha = float(max(0.0, min(1.0, 1.0 - self.fallback_beta)))

        for _layer, items in layer_groups.items():
            a_sum = 0.0
            b_sum = 0.0
            c_sum = 0.0
            can_dynamic = True
            for name, param in items:
                if name not in ema_bank:
                    ema_bank[name] = param.detach().to("cpu")
                theta = param.detach().float().cpu()
                ema = ema_bank[name].detach().float().cpu()
                a_sum += float((theta - ema).abs().sum().item())
                grad = grad_snap.get(name)
                if grad is None:
                    can_dynamic = False
                    continue
                g = grad.detach().float().cpu()
                b_sum += float(g.abs().sum().item())
                prev = prev_grads.get(name)
                if prev is None:
                    can_dynamic = False
                    continue
                c_sum += float((g - prev.detach().float().cpu()).abs().sum().item())

            alpha = fallback_alpha
            used_fallback = True
            if can_dynamic and a_sum > 0.0 and c_sum > 0.0:
                try:
                    raw_alpha = float((1.0 + b_sum) / (a_sum * c_sum) * 2.0)
                    if 0.0 < raw_alpha < 1.0:
                        alpha = raw_alpha
                        used_fallback = False
                except Exception:
                    alpha = fallback_alpha
                    used_fallback = True
            if used_fallback:
                fallbacks += 1
            beta = 1.0 - float(alpha)
            betas.append(beta)

            with torch.no_grad():
                for name, param in items:
                    if name not in ema_bank:
                        ema_bank[name] = param.detach().to("cpu")
                    shadow = ema_bank[name].detach().to(dtype=param.dtype)
                    p_cpu = param.detach().to(device="cpu", dtype=param.dtype)
                    shadow.mul_(beta).add_(p_cpu, alpha=alpha)
                    ema_bank[name] = shadow if not self.cpu_offload_ema else shadow.cpu()
                    updated += 1

        for name, grad in grad_snap.items():
            prev_grads[name] = grad.detach().float().cpu() if torch.is_tensor(grad) else grad
        return {
            "updated": int(updated),
            "fallbacks": int(fallbacks),
            "beta_mean": (sum(betas) / max(1, len(betas))) if betas else None,
        }

    def _copy_ema_into_adapter(self, lora: Any, adapter_name: str) -> None:
        ema_bank = self._ema.get(adapter_name) or {}
        if not ema_bank:
            return
        with torch.no_grad():
            for name, param in _iter_adapter_params(lora, adapter_name):
                shadow = ema_bank.get(name)
                if shadow is None:
                    continue
                param.data.copy_(shadow.data.to(device=param.device, dtype=param.dtype))


def _method_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    advanced_cfg = cfg.get("advanced_baseline", {}) if isinstance(cfg.get("advanced_baseline", {}), dict) else {}
    nested = advanced_cfg.get(name, {}) if isinstance(advanced_cfg.get(name, {}), dict) else {}
    top = cfg.get(name, {}) if isinstance(cfg.get(name, {}), dict) else {}
    merged = dict(nested)
    merged.update(top)
    return merged


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


def _unique_instructions(examples: Sequence[Example]) -> List[str]:
    seen = set()
    out: List[str] = []
    for ex in examples:
        text = str(getattr(ex, "instruction", "") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _iter_adapter_params(lora: Any, adapter_name: str):
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is None:
        return
    needle_a = f"lora_A.{adapter_name}."
    needle_b = f"lora_B.{adapter_name}."
    for name, param in peft_model.named_parameters():
        if needle_a in name or needle_b in name:
            yield name, param


def _lora_compute_device(lora: Any):
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is None:
        return None
    for param in peft_model.parameters():
        if param.device.type == "cuda":
            return param.device
    try:
        return next(peft_model.parameters()).device
    except StopIteration:
        return None


def _move_adapter_params(lora: Any, adapter_name: str, device: Any) -> None:
    if device is None:
        return
    with torch.no_grad():
        for _, param in _iter_adapter_params(lora, adapter_name):
            if param.device != device:
                param.data = param.data.to(device, non_blocking=False)


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _layer_key(param_name: str, adapter_name: str) -> str:
    key = param_name
    for tag in (f"lora_A.{adapter_name}.", f"lora_B.{adapter_name}."):
        key = key.replace(tag, "")
    for suffix in (".weight", ".bias"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def _tokenize(text: str) -> List[str]:
    raw = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or ""))
    return [tok for tok in raw.split() if tok]


_EN_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "not",
    "no",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "may",
    "into",
    "than",
    "then",
    "so",
    "such",
    "only",
    "own",
    "same",
    "too",
    "very",
}


def _mean_pairwise_tfidf_cosine(
    current: Sequence[str],
    historical: Sequence[str],
    *,
    stop_words: str = "english",
) -> float:
    if not current or not historical:
        return 0.0
    sims: List[float] = []
    for cur in current:
        for hist in historical:
            sims.append(_tfidf_cosine_pair(str(cur), str(hist), stop_words=stop_words))
    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


def _tfidf_cosine_pair(a: str, b: str, *, stop_words: str = "english") -> float:
    if not str(a).strip() or not str(b).strip():
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        sw: Any = "english" if str(stop_words).lower() == "english" else None
        vectorizer = TfidfVectorizer(stop_words=sw)
        matrix = vectorizer.fit_transform([str(a), str(b)])
        return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    except Exception:
        return _tfidf_cosine_fallback(str(a), str(b), stop_words=stop_words)


def _tfidf_cosine_fallback(a: str, b: str, *, stop_words: str = "english") -> float:
    stop = _EN_STOP if str(stop_words).lower() == "english" else set()
    tok_a = [t for t in _tokenize(a) if t not in stop]
    tok_b = [t for t in _tokenize(b) if t not in stop]
    if not tok_a or not tok_b:
        return 0.0
    docs = [tok_a, tok_b]
    vocab = sorted(set(tok_a) | set(tok_b))
    df = {term: sum(1 for doc in docs if term in doc) for term in vocab}
    import math

    def _vec(tokens: List[str]) -> List[float]:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        n = max(1, len(tokens))
        out: List[float] = []
        for term in vocab:
            idf = math.log((1.0 + 2.0) / (1.0 + df[term])) + 1.0
            out.append((tf.get(term, 0) / n) * idf)
        return out

    va, vb = _vec(tok_a), _vec(tok_b)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(dot / (na * nb))
