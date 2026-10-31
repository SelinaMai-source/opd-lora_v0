from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from core.data import Example, Segment


ReplayPair = Tuple[Tuple[str, str], str]


def method_cfg(cfg: Dict[str, Any], name: str) -> Dict[str, Any]:
    nested = cfg.get("citb", {}) if isinstance(cfg.get("citb", {}), dict) else {}
    top = cfg.get(name, {}) if isinstance(cfg.get(name, {}), dict) else {}
    merged = dict(nested)
    merged.update(top)
    return merged


def replay_k(cfg: Dict[str, Any], default: int = 50) -> int:
    citb = method_cfg(cfg, "citb")
    replay = cfg.get("replay", {}) if isinstance(cfg.get("replay", {}), dict) else {}
    if "k" in replay:
        return max(0, int(replay.get("k", default)))
    if "k" in citb:
        return max(0, int(citb.get("k", default)))
    if "memory_k" in citb:
        return max(0, int(citb.get("memory_k", default)))
    return max(0, int(default))


def reg_lambda(cfg: Dict[str, Any], default: float = 0.01) -> float:
    citb = method_cfg(cfg, "citb")
    return float(citb.get("reg", default))


def cfg_seed(cfg: Dict[str, Any], default: int = 456) -> int:
    return int(cfg.get("seed", default))


def to_pairs(
    examples: Sequence[Example],
    *,
    strip_instruction: bool = False,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    pairs: List[Tuple[str, str]] = []
    targets: List[str] = []
    for ex in examples:
        instruction = "" if strip_instruction else ex.instruction
        pairs.append((instruction, ex.input))
        targets.append(ex.output)
    return pairs, targets


def batch_pairs(
    pairs: List[Tuple[str, str]],
    targets: List[str],
    batch_size: int,
) -> Iterable[Tuple[List[Tuple[str, str]], List[str]]]:
    bs = max(1, int(batch_size))
    for i in range(0, len(pairs), bs):
        yield pairs[i : i + bs], targets[i : i + bs]


def empty_train_metrics(*, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "batches": 0,
        "mean_batch_acc": 0.0,
        "train.loss": 0.0,
        "train.answer_token_acc": 0.0,
        "num_total_tokens": 0,
        "num_supervised_tokens": 0,
        "grad_norm": 0.0,
        "lr": 0.0,
        "skipped_train": True,
    }
    if extra:
        metrics.update(extra)
    return metrics


def summarize_sft_metrics(
    *,
    batches: int,
    batch_accs: List[float],
    batch_losses: List[float],
    answer_token_accs: List[float],
    grad_norms: List[float],
    lr_values: List[float],
    total_tokens: int,
    supervised_tokens: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "batches": int(batches),
        "mean_batch_acc": sum(batch_accs) / max(1, len(batch_accs)),
        "train.loss": sum(batch_losses) / max(1, len(batch_losses)),
        "train.answer_token_acc": sum(answer_token_accs) / max(1, len(answer_token_accs)),
        "num_total_tokens": int(total_tokens),
        "num_supervised_tokens": int(supervised_tokens),
        "grad_norm": sum(grad_norms) / max(1, len(grad_norms)),
        "lr": sum(lr_values) / max(1, len(lr_values)),
        "skipped_train": False,
    }
    if extra:
        metrics.update(extra)
    return metrics


def run_ce_sft(
    *,
    model: Any,
    lora: Any,
    pairs: List[Tuple[str, str]],
    targets: List[str],
    lr: float,
    epochs: int,
    batch_size: int,
    rng: Optional[random.Random] = None,
    shuffle: bool = True,
) -> Dict[str, Any]:
    batch_accs: List[float] = []
    batch_losses: List[float] = []
    answer_token_accs: List[float] = []
    grad_norms: List[float] = []
    lr_values: List[float] = []
    total_tokens = 0
    supervised_tokens = 0
    batches = 0
    for _ in range(max(1, int(epochs))):
        order_pairs, order_targets = pairs, targets
        if shuffle and rng is not None and len(pairs) > 1:
            idxs = list(range(len(pairs)))
            rng.shuffle(idxs)
            order_pairs = [pairs[i] for i in idxs]
            order_targets = [targets[i] for i in idxs]
        for b_pairs, b_targets in batch_pairs(order_pairs, order_targets, batch_size):
            out = model.fit_batch(b_pairs, b_targets, lr=lr)
            step_stats = lora.step_adapter()
            batch_accs.append(float(out.get("train_batch_acc", 0.0)))
            batch_losses.append(float(out.get("train_loss", 0.0)))
            answer_token_accs.append(float(out.get("train_answer_token_acc", 0.0)))
            grad_norms.append(float(step_stats.get("grad_norm", 0.0)))
            lr_values.append(float(step_stats.get("lr", lr)))
            total_tokens += int(out.get("num_total_tokens", 0))
            supervised_tokens += int(out.get("num_supervised_tokens", 0))
            batches += 1
    return summarize_sft_metrics(
        batches=batches,
        batch_accs=batch_accs,
        batch_losses=batch_losses,
        answer_token_accs=answer_token_accs,
        grad_norms=grad_norms,
        lr_values=lr_values,
        total_tokens=total_tokens,
        supervised_tokens=supervised_tokens,
    )


def sample_k(
    pairs: List[Tuple[str, str]],
    targets: List[str],
    k: int,
    rng: random.Random,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    n = len(pairs)
    if n <= 0 or k <= 0:
        return [], []
    take = min(int(k), n)
    idxs = list(range(n))
    rng.shuffle(idxs)
    chosen = idxs[:take]
    return [pairs[i] for i in chosen], [targets[i] for i in chosen]


def ensure_default_adapter(lora: Any) -> str:
    if hasattr(lora, "list_adapters") and "default" in lora.list_adapters():
        lora.set_active_adapter("default")
    return lora.get_active_adapter_name() if hasattr(lora, "get_active_adapter_name") else "default"


def zero_lora_grad(lora: Any) -> None:
    opt = getattr(lora, "_optimizer", None)
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is not None:
        peft_model.zero_grad(set_to_none=True)


def iter_trainable_lora_params(lora: Any) -> List[Tuple[str, Any]]:
    peft_model = getattr(lora, "peft_model", None)
    if peft_model is None:
        return []
    out: List[Tuple[str, Any]] = []
    for name, param in peft_model.named_parameters():
        if "lora_" in name and param.requires_grad:
            out.append((name, param))
    return out


def snapshot_trainable(lora: Any) -> Dict[str, torch.Tensor]:
    snapped: Dict[str, torch.Tensor] = {}
    for name, param in iter_trainable_lora_params(lora):
        snapped[name] = param.detach().float().cpu().clone()
    return snapped


def flatten_grads(params: Sequence[Tuple[str, Any]]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    device = None
    for _, param in params:
        if device is None:
            device = param.device
        if param.grad is None:
            parts.append(torch.zeros(param.numel(), device=device, dtype=torch.float32))
        else:
            parts.append(param.grad.detach().float().reshape(-1))
    if not parts:
        return torch.zeros(1, dtype=torch.float32)
    return torch.cat(parts)


def assign_grads(params: Sequence[Tuple[str, Any]], flat: torch.Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for _, param in params:
            n = int(param.numel())
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.copy_(flat[offset : offset + n].reshape_as(param).to(dtype=param.dtype))
            offset += n


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def segment_train_examples(stream: Any) -> List[Example]:
    examples: List[Example] = []
    segments = getattr(stream, "stream", None) or getattr(stream, "segments", None) or []
    for seg in segments:
        examples.extend(list(getattr(seg, "train", []) or []))
    return examples


def set_include_instruction(model: Any, *, include: bool) -> None:
    model.train_include_instruction = bool(include)
    model.eval_include_instruction = bool(include)
