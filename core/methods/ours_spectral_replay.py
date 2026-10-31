"""Spectral Sparse Replay Gating (SSRG) for Ours v10.

Novelty: replay sample selection via truncated SVD of hidden-state covariance —
retain only top spectral components that explain >= energy_threshold variance,
then gate replay by projection energy (sparse spectral subspace matching).
Distinct from uniform replay (Replay LoRA) and prompt-pool methods (PP).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from core.data import Example, Segment


@dataclass
class _ReplayItem:
    instruction: str
    input_text: str
    target: str
    segment_id: int


class SpectralSparseReplayGate:
  def __init__(self, cfg: Dict[str, Any]):
    self.buffer_size = int(cfg.get("buffer_size", 256))
    self.replay_ratio = float(cfg.get("replay_ratio", 0.25))
    self.spectral_top_k = int(cfg.get("spectral_top_k", 8))
    self.spectral_energy_threshold = float(cfg.get("spectral_energy_threshold", 0.85))
    self.min_replay_per_segment = int(cfg.get("min_replay_per_segment", 4))
    self._buffer: List[_ReplayItem] = []
    self._last_metrics: Dict[str, Any] = {}

  def state_dict(self) -> Dict[str, Any]:
    return {
      "buffer_size": len(self._buffer),
      "last_metrics": dict(self._last_metrics),
    }

  def ingest_segment(self, segment: Segment) -> None:
    for ex in segment.train:
      self._buffer.append(
        _ReplayItem(
          instruction=ex.instruction,
          input_text=ex.input,
          target=ex.output,
          segment_id=segment.segment_id,
        )
      )
    if len(self._buffer) > self.buffer_size:
      self._buffer = self._buffer[-self.buffer_size :]

  def augment_segment(
    self,
    segment: Segment,
    *,
    model: Any,
    prompt_fn,
  ) -> Tuple[Segment, Dict[str, Any]]:
    """Return segment with spectrally-gated replay examples mixed into train."""
    metrics: Dict[str, Any] = {
      "ssrg_buffer_size": len(self._buffer),
      "ssrg_replay_added": 0,
      "ssrg_spectral_rank": 0,
      "ssrg_mean_gate_score": 0.0,
    }
    if segment.segment_id == 0 or not self._buffer or self.replay_ratio <= 0:
      self._last_metrics = metrics
      return segment, metrics

    current_n = len(segment.train)
    num_replay = max(
      self.min_replay_per_segment,
      int(round(current_n * self.replay_ratio)),
    )
    num_replay = min(num_replay, len(self._buffer))
    if num_replay <= 0:
      self._last_metrics = metrics
      return segment, metrics

    gated = self._spectral_gate_select(
      model=model,
      prompt_fn=prompt_fn,
      k=num_replay,
    )
    if not gated:
      gated = random.sample(self._buffer, k=num_replay)

    replay_examples = [
      Example(
        instruction=it.instruction,
        input=it.input_text,
        output=it.target,
      )
      for it in gated
    ]
    merged_train = list(segment.train) + replay_examples
    aug = Segment(
      segment_id=segment.segment_id,
      segment_name=segment.segment_name,
      train=merged_train,
      eval=segment.eval,
    )
    metrics["ssrg_replay_added"] = len(replay_examples)
    metrics["ssrg_mean_gate_score"] = float(
      sum(getattr(it, "_gate_score", 1.0) for it in gated) / max(1, len(gated))
    )
    self._last_metrics = metrics
    return aug, metrics

  def _spectral_gate_select(
    self,
    *,
    model: Any,
    prompt_fn,
    k: int,
  ) -> List[_ReplayItem]:
    if not hasattr(model, "get_activations"):
      return random.sample(self._buffer, k=min(k, len(self._buffer)))

    sample_cap = min(64, len(self._buffer))
    candidates = random.sample(self._buffer, k=sample_cap) if len(self._buffer) > sample_cap else list(self._buffer)
    prompts = [prompt_fn(it.instruction, it.input_text) for it in candidates]
    try:
      acts = model.get_activations(prompts)
    except Exception:
      return random.sample(self._buffer, k=min(k, len(self._buffer)))

    mat = []
    for row in acts:
      if not row:
        continue
      t = torch.tensor(row, dtype=torch.float32)
      if t.dim() > 1:
        t = t.mean(dim=0)
      mat.append(t)
    if len(mat) < 2:
      return random.sample(self._buffer, k=min(k, len(self._buffer)))

    X = torch.stack(mat, dim=0)
    X = X - X.mean(dim=0, keepdim=True)
    cov = (X.T @ X) / max(1, X.shape[0] - 1)
    try:
      u, s, _ = torch.linalg.svd(cov, full_matrices=False)
    except RuntimeError:
      return random.sample(self._buffer, k=min(k, len(self._buffer)))

    energy = (s ** 2).cumsum(0) / (s ** 2).sum().clamp(min=1e-8)
    rank = int((energy >= self.spectral_energy_threshold).nonzero(as_tuple=True)[0][:1].numel())
    rank = max(1, min(rank if rank > 0 else 1, self.spectral_top_k, s.numel()))
    basis = u[:, :rank]

    scores: List[Tuple[float, _ReplayItem]] = []
    for vec, item in zip(X, candidates):
      proj = basis.T @ vec
      score = float((proj ** 2).sum().sqrt().item())
      item._gate_score = score  # type: ignore[attr-defined]
      scores.append((score, item))
    scores.sort(key=lambda x: x[0], reverse=True)
    self._last_metrics["ssrg_spectral_rank"] = rank
    top = [it for _, it in scores[:k]]
    return top
