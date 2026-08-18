"""
Causal LM training metrics aligned with HuggingFace `labels` handling.

HuggingFace causal LMs compute loss on shifted positions:
  shift_logits = logits[..., :-1, :]
  shift_labels = labels[..., 1:]
A token at `labels[b, t]` is predicted from `logits[b, t-1]`, not `logits[b, t]`.

Comparing `argmax(logits[b, t])` to `labels[b, t]` (same index) is **not** equivalent
to the cross-entropy terms that contribute to `outputs.loss`.
"""

from __future__ import annotations

from typing import Tuple

import torch


def causal_lm_shifted_logits_argmax(logits: torch.Tensor) -> torch.Tensor:
    """Argmax at each position that pairs with `labels[:, 1:]`. Shape [B, T-1]."""
    return logits[:, :-1, :].argmax(dim=-1)


def causal_lm_shifted_labels(labels: torch.Tensor) -> torch.Tensor:
    """Label tensor aligned with shifted logits. Shape [B, T-1]."""
    return labels[:, 1:].contiguous()


def shifted_supervised_mask(labels: torch.Tensor) -> torch.Tensor:
    """Boolean mask where a causal-LM loss term is computed (shifted label != -100)."""
    return causal_lm_shifted_labels(labels).ne(-100)


def teacher_forced_token_accuracy_shifted(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, int, int]:
    """
    Teacher-forced token accuracy matching HF causal LM loss span.

    Returns:
      (accuracy, num_correct, num_loss_tokens)
    """
    shift_pred = causal_lm_shifted_logits_argmax(logits)
    shift_lab = causal_lm_shifted_labels(labels)
    mask = shift_lab.ne(-100)
    correct = int(((shift_pred == shift_lab) & mask).sum().item())
    n_loss = int(mask.sum().item())
    return float(correct / max(1, n_loss)), correct, n_loss


def count_supervised_label_tokens(labels: torch.Tensor) -> int:
    """Count of positions where `labels != -100` (includes the first token slot)."""
    return int(labels.ne(-100).sum().item())
