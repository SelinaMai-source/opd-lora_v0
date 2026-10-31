from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from core.data import Example, Segment

from baselines.citb.common import (
    batch_pairs,
    cfg_seed,
    empty_cuda_cache,
    ensure_default_adapter,
    flatten_grads,
    iter_trainable_lora_params,
    run_ce_sft,
    sample_k,
    summarize_sft_metrics,
    to_pairs,
    zero_lora_grad,
)

ReplayPair = Tuple[Tuple[str, str], str]


def method_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    nested = cfg.get("lit", {}) if isinstance(cfg.get("lit", {}), dict) else {}
    top = cfg.get(name, {}) if isinstance(cfg.get(name, {}), dict) else {}
    merged = dict(nested)
    merged.update(top)
    return merged


def replay_k(cfg: Dict[str, Any], name: str, default: int = 10) -> int:
    block = method_cfg(cfg, name)
    replay = cfg.get("replay", {}) if isinstance(cfg.get("replay", {}), dict) else {}
    if "k" in block:
        return max(0, int(block.get("k", default)))
    if "memory_k" in block:
        return max(0, int(block.get("memory_k", default)))
    if "k" in replay:
        return max(0, int(replay.get("k", default)))
    if "buffer_k" in replay:
        return max(0, int(replay.get("buffer_k", default)))
    return max(0, int(default))


def unique_instructions(examples: Sequence[Example]) -> List[str]:
    seen = set()
    out: List[str] = []
    for ex in examples:
        ins = str(getattr(ex, "instruction", "") or "").strip()
        if ins and ins not in seen:
            seen.add(ins)
            out.append(ins)
    return out


def instruction_of(pair: Tuple[str, str]) -> str:
    return str(pair[0] or "").strip()


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize_text(text: str) -> List[str]:
    toks = _TOKEN_RE.findall(str(text or "").lower())
    return toks or [str(text or "").strip().lower() or "<empty>"]


def hashed_bow(texts: Sequence[str], dim: int = 256) -> torch.Tensor:
    """Deterministic hashed bag-of-words embeddings (no extra encoder)."""
    dim = max(8, int(dim))
    rows = []
    for text in texts:
        vec = torch.zeros(dim, dtype=torch.float32)
        toks = tokenize_text(text)
        for tok in toks:
            h = hash(tok) % dim
            vec[h] += 1.0
        n = float(vec.norm().item())
        if n > 0:
            vec /= n
        rows.append(vec)
    if not rows:
        return torch.zeros(0, dim, dtype=torch.float32)
    return torch.stack(rows, dim=0)


def mean_embedding(texts: Sequence[str], dim: int = 256) -> torch.Tensor:
    mat = hashed_bow(texts, dim=dim)
    if mat.numel() == 0:
        return torch.zeros(dim, dtype=torch.float32)
    v = mat.mean(dim=0)
    n = float(v.norm().item())
    if n > 0:
        v = v / n
    return v


def cosine_distance_mat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = torch.nn.functional.normalize(a, dim=-1, eps=1e-8)
    b_n = torch.nn.functional.normalize(b, dim=-1, eps=1e-8)
    return (1.0 - (a_n @ b_n.T)).clamp(min=0.0)


def wasserstein_uniform(a: torch.Tensor, b: torch.Tensor) -> float:
    """1-Wasserstein with cosine cost. Uses POT if present, else mean pairwise cost."""
    if a.numel() == 0 or b.numel() == 0:
        return 1.0
    aa = a.detach().float().cpu()
    bb = b.detach().float().cpu()
    try:
        import ot  # type: ignore

        cost = cosine_distance_mat(aa, bb).numpy()
        alpha = ot.unif(int(aa.shape[0]))
        beta = ot.unif(int(bb.shape[0]))
        return float(ot.emd2(alpha, beta, cost, numItermax=100000))
    except Exception:
        cost = cosine_distance_mat(aa, bb)
        return float(cost.mean().item())


def insinfo_score(instruction: str) -> float:
    """InsInfo proxy without GPT-4 tags: length * unique-token diversity."""
    toks = tokenize_text(instruction)
    n = max(1, len(toks))
    uniq = len(set(toks))
    return float(math.log1p(n) * (uniq / n))


def mask_key_parts(instruction: str, frac: float = 0.3) -> str:
    """Drop the longest alphabetic tokens (instruction key-part proxy)."""
    toks = str(instruction or "").split()
    if not toks:
        return str(instruction or "")
    n_drop = max(1, int(round(len(toks) * max(0.0, min(0.8, float(frac))))))
    ranked = sorted(range(len(toks)), key=lambda i: len(re.sub(r"[^A-Za-z0-9]", "", toks[i])), reverse=True)
    drop = set(ranked[:n_drop])
    kept = [t for i, t in enumerate(toks) if i not in drop]
    return " ".join(kept) if kept else toks[0]


def allocate_by_wasserstein(
    current_emb: torch.Tensor,
    past_embs: Sequence[torch.Tensor],
    total: int,
) -> List[int]:
    if total <= 0 or not past_embs:
        return [0] * len(past_embs)
    dists = [max(1e-6, wasserstein_uniform(current_emb, pe)) for pe in past_embs]
    s = sum(dists)
    raw = [int(total * d / s) for d in dists]
    while sum(raw) < total:
        raw[dists.index(max(dists))] += 1
        dists[dists.index(max(dists))] *= 0.999
    while sum(raw) > total:
        j = max(range(len(raw)), key=lambda i: raw[i])
        if raw[j] <= 0:
            break
        raw[j] -= 1
    return raw


def sample_by_insinfo(
    pairs: List[Tuple[str, str]],
    targets: List[str],
    k: int,
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    if k <= 0 or not pairs:
        return [], []
    scored = []
    for i, (p, t) in enumerate(zip(pairs, targets)):
        scored.append((insinfo_score(instruction_of(p)), i))
    scored.sort(key=lambda x: x[0], reverse=True)
    take = min(int(k), len(scored))
    # Keep top-InsInfo instructions, then random among those examples.
    top_idx = [i for _, i in scored[: max(take, 1)]]
    rng.shuffle(top_idx)
    chosen = top_idx[:take]
    return [pairs[i] for i in chosen], [targets[i] for i in chosen]


def prepare_train_adapter(lora: Any, name: str, model: Any = None) -> str:
    """Clear eval routing / disable_adapters and train only `name`.

    `fit_batch` calls `generate` for train-acc, and eval generate can leave PEFT
    layers with `disable_adapters` or leftover `_soft_routing_adapters`. The
    next backward then has no grad_fn. Always call this before CE/OPSD steps.
    """
    peft_model = getattr(lora, "peft_model", None)
    if hasattr(lora, "set_soft_routing"):
        lora.set_soft_routing(None, None)
    if peft_model is not None:
        if hasattr(peft_model, "enable_adapter_layers"):
            try:
                peft_model.enable_adapter_layers()
            except Exception:
                pass
        import peft.tuners.lora.layer as lora_layer

        for module in peft_model.modules():
            if isinstance(module, lora_layer.Linear):
                object.__setattr__(module, "_soft_routing_adapters", None)
    if hasattr(lora, "unfreeze_adapter"):
        lora.unfreeze_adapter(name)
    lora.set_active_adapter(name)
    if peft_model is not None:
        frozen = set(getattr(lora, "_frozen_adapters", set()) or set())
        for param_name, param in peft_model.named_parameters():
            if "lora_" not in param_name:
                param.requires_grad_(False)
                continue
            want = (
                f"lora_A.{name}." in param_name
                or f"lora_B.{name}." in param_name
                or f".{name}." in param_name
            )
            param.requires_grad_(bool(want and name not in frozen))
    inner = getattr(model, "model", None) if model is not None else None
    if inner is not None and hasattr(inner, "train"):
        inner.train()
    setattr(lora, "_train_adapter_name", name)
    return lora.get_active_adapter_name() if hasattr(lora, "get_active_adapter_name") else name


def spawn_segment_adapter(lora: Any, name: str, *, freeze_others: bool = True) -> str:
    if name not in lora.list_adapters():
        lora.create_adapter(name)
    if freeze_others and hasattr(lora, "freeze_adapter"):
        for old in lora.list_adapters():
            if old != name:
                lora.freeze_adapter(old)
    lora.set_active_adapter(name)
    if hasattr(lora, "unfreeze_adapter"):
        lora.unfreeze_adapter(name)
    setattr(lora, "_train_adapter_name", name)
    return lora.get_active_adapter_name()


def install_oracle_eval_hook(method: Any, model: Any, lookup) -> None:
    """Route eval to the adapter that trained the source segment (AdapterCL-style)."""

    def _on_eval_segment(segment: Segment) -> None:
        lora = getattr(method, "_lora", None)
        if lora is None:
            return
        name = lookup(segment)
        if not name:
            return
        adapters = set(lora.list_adapters()) if hasattr(lora, "list_adapters") else set()
        if name not in adapters:
            return
        lora.set_active_adapter(name)

    model.on_eval_segment = _on_eval_segment


def wrap_generate_router(method: Any, model: Any, route_fn) -> None:
    if getattr(method, "_eval_wrapped", False):
        return
    orig_generate = model.generate
    orig_ids = getattr(model, "generate_with_ids", None)

    def _restore_train_adapter() -> None:
        lora = getattr(method, "_lora", None)
        if lora is None:
            return
        prev = getattr(lora, "_train_adapter_name", None) or (
            lora.get_active_adapter_name() if hasattr(lora, "get_active_adapter_name") else None
        )
        if prev and hasattr(lora, "list_adapters") and prev in lora.list_adapters():
            prepare_train_adapter(lora, prev, model)
        elif hasattr(lora, "set_soft_routing"):
            lora.set_soft_routing(None, None)

    def _generate(prompts: List[str], *args: Any, **kwargs: Any):
        route_fn(prompts)
        try:
            return orig_generate(prompts, *args, **kwargs)
        finally:
            _restore_train_adapter()

    model.generate = _generate
    if callable(orig_ids):

        def _generate_ids(*args: Any, **kwargs: Any):
            prompts = kwargs.get("prompts") or (args[0] if args else None)
            if isinstance(prompts, list):
                route_fn(prompts)
            try:
                return orig_ids(*args, **kwargs)
            finally:
                _restore_train_adapter()

        model.generate_with_ids = _generate_ids
    method._eval_wrapped = True


def lora_named_params(lora: Any, adapter_name: Optional[str] = None) -> List[Tuple[str, Any]]:
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is None:
        return []
    out: List[Tuple[str, Any]] = []
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            continue
        if adapter_name is not None and f".{adapter_name}." not in name:
            continue
        out.append((name, param))
    return out


def snapshot_adapter(lora: Any, adapter_name: str) -> Dict[str, torch.Tensor]:
    snapped: Dict[str, torch.Tensor] = {}
    for name, param in lora_named_params(lora, adapter_name):
        snapped[name] = param.detach().float().cpu().clone()
    return snapped


def diversity_select(
    pairs: List[Tuple[str, str]],
    targets: List[str],
    k: int,
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Greedy farthest-first on hashed instruction+output embeddings."""
    n = len(pairs)
    if k <= 0 or n <= 0:
        return [], []
    take = min(int(k), n)
    texts = [f"{p[0]}\n{p[1]}\n{t}" for p, t in zip(pairs, targets)]
    emb = hashed_bow(texts, dim=128)
    start = rng.randrange(n)
    chosen = [start]
    remaining = [i for i in range(n) if i != start]
    while len(chosen) < take and remaining:
        sel = torch.stack([emb[i] for i in chosen], dim=0)
        rest = torch.stack([emb[i] for i in remaining], dim=0)
        dist = cosine_distance_mat(rest, sel).min(dim=1).values
        j = int(torch.argmax(dist).item())
        chosen.append(remaining.pop(j))
    return [pairs[i] for i in chosen], [targets[i] for i in chosen]


def mix_current_and_memory(
    current_pairs: List[Tuple[str, str]],
    current_targets: List[str],
    memory_pairs: List[Tuple[str, str]],
    memory_targets: List[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    return list(current_pairs) + list(memory_pairs), list(current_targets) + list(memory_targets)
