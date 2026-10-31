from __future__ import annotations

"""GainLoRA (NeurIPS 2025): one new LoRA per segment + gate on the new adapter.

Training uses only the new adapter. Eval keeps the nearest frozen adapter at
weight 1 and gates the new adapter down when the prompt is closer to an old
centroid (`set_soft_routing`). Do not sum every historical adapter at once.
"""

import random
from typing import Any, Dict, List, Tuple

import torch

from core.data import Segment
from core.formatting import format_for_infer

from baselines.citb.common import run_ce_sft, to_pairs
from baselines.lit_common import (
    cfg_seed,
    hashed_bow,
    mean_embedding,
    method_cfg,
    prepare_train_adapter,
    spawn_segment_adapter,
    wrap_generate_router,
)


class GainLoRAMethod:
    name = "gainlora"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "gainlora")
        self.adapter_prefix = str(block.get("adapter_prefix", "gain_s"))
        self.old_gate = float(block.get("old_task_new_gate", 0.05))
        self._rng = random.Random(cfg_seed(cfg))
        self._adapters: List[str] = []
        self._centroids: Dict[str, torch.Tensor] = {}
        self._lora: Any = None
        self._eval_wrapped = False

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        self._lora = lora
        self._install_router(model)
        name = f"{self.adapter_prefix}{int(segment.segment_id)}"
        spawn_segment_adapter(lora, name, freeze_others=True)
        if name not in self._adapters:
            self._adapters.append(name)
        if hasattr(lora, "set_soft_routing"):
            lora.set_soft_routing(None, None)
        return {
            "active_adapter": lora.get_active_adapter_name(),
            "num_adapters": len(self._adapters),
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
        name = lora.get_active_adapter_name()
        if name:
            prepare_train_adapter(lora, name, model)
        pairs, targets = to_pairs(segment.train)
        metrics = run_ce_sft(
            model=model,
            lora=lora,
            pairs=pairs,
            targets=targets,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            rng=self._rng,
            shuffle=True,
        )
        active = lora.get_active_adapter_name()
        tok = getattr(model, "tokenizer", None)
        if tok is not None and pairs:
            centroid_texts = [format_for_infer(tok, p[0], p[1]) for p in pairs]
        else:
            centroid_texts = [f"{p[0]}\n{p[1]}" for p in pairs] or [str(segment.segment_name)]
        self._centroids[active] = mean_embedding(centroid_texts)
        metrics["active_adapter"] = active
        metrics["num_adapters"] = len(self._adapters)
        return metrics

    def _install_router(self, model: Any) -> None:
        def _route(prompts: List[str]) -> None:
            lora = self._lora
            if lora is None or not self._adapters or not hasattr(lora, "set_soft_routing"):
                return
            current = self._adapters[-1]
            if len(self._adapters) == 1:
                lora.set_soft_routing(None, None)
                lora.set_active_adapter(current)
                return
            texts = prompts if prompts else [""]
            q = hashed_bow(texts, dim=256).mean(dim=0)
            q = torch.nn.functional.normalize(q, dim=0, eps=1e-8)
            best_name = current
            best_sim = -1e9
            for name in self._adapters:
                c = self._centroids.get(name)
                if c is None:
                    continue
                sim = float(torch.dot(q, torch.nn.functional.normalize(c, dim=0, eps=1e-8)).item())
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            # Official GainLoRA gate: keep the matched frozen adapter at 1.0 and
            # only scale the *new* LoRA down on old-task prompts. Summing every
            # historical adapter at weight 1.0 collapses generation.
            if best_name == current:
                lora.set_soft_routing([current], [1.0])
                lora.set_active_adapter(current)
                return
            lora.set_soft_routing([best_name, current], [1.0, self.old_gate])
            lora.set_active_adapter(best_name)

        wrap_generate_router(self, model, _route)
