"""On-policy distillation: student rollout, frozen base teacher, JSD, added to CE.

The teacher forward used in training is ``baselines.basic_baselines.opsd.method._teacher_forward_logits``
with ``teacher_source='base'``, which runs under ``torch.no_grad`` and PEFT
``disable_adapter()``. This module owns the divergence so that formula is not
duplicated. CE is added by the caller as ``L = L_CE + λ L_OPSD``.
"""
from __future__ import annotations

from typing import Any


def token_divergence(teacher_logits: Any, student_logits: Any, *, kind: str, beta: float) -> Any:
    """Mean per-token divergence D(p_T || p_S) over the rollout span.

    ``teacher_logits`` and ``student_logits`` are ``[R, V]``. JSD uses mixture
    weight ``beta`` on the teacher. KL is ``KL(p_T || p_S)``.
    """
    import torch
    import torch.nn.functional as F

    log_t = F.log_softmax(teacher_logits, dim=-1)
    log_s = F.log_softmax(student_logits, dim=-1)
    p_t = log_t.exp()
    p_s = log_s.exp()
    if kind == "kl":
        per_token = (p_t * (log_t - log_s)).sum(dim=-1)
    elif kind == "jsd":
        b = min(max(float(beta), 1e-6), 1.0 - 1e-6)
        log_m = torch.log(b * p_t + (1.0 - b) * p_s + 1e-12)
        kl_t = (p_t * (log_t - log_m)).sum(dim=-1)
        kl_s = (p_s * (log_s - log_m)).sum(dim=-1)
        per_token = b * kl_t + (1.0 - b) * kl_s
    else:
        raise ValueError(f"divergence kind must be 'jsd' or 'kl', got: {kind}")
    return per_token.mean()


def ce_plus_opsd(ce_loss: Any, opsd_loss: Any, lambda_opsd: float) -> Any:
    """Same-step objective. ``lambda_opsd <= 0`` leaves CE unchanged."""
    if float(lambda_opsd) <= 0.0 or opsd_loss is None:
        return ce_loss
    return ce_loss + float(lambda_opsd) * opsd_loss
