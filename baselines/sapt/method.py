from __future__ import annotations

"""SAPT (ACL 2024): per-segment LoRA + shared-attention routing + ARM pseudo samples.

SALS: softmax over instruction-centroid cosine as `set_soft_routing` weights.
ARM: generate K=10 pseudo (instruction, input)→output from the current model
on previous-segment prompts, then mix into CE. Eval uses the same attention
routing. Official T5/Llama trainers are not run.
"""

import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from core.data import Segment
from core.formatting import format_for_infer

from baselines.citb.common import run_ce_sft, to_pairs
from baselines.lit_common import (
    cfg_seed,
    hashed_bow,
    mean_embedding,
    method_cfg,
    prepare_train_adapter,
    replay_k,
    spawn_segment_adapter,
    wrap_generate_router,
)


class SAPTMethod:
    name = "sapt"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "sapt")
        self.adapter_prefix = str(block.get("adapter_prefix", "sapt_s"))
        self.k = replay_k(cfg, "sapt", default=10)
        self.attn_temp = float(block.get("attn_temp", 0.1))
        self._rng = random.Random(cfg_seed(cfg))
        self._adapters: List[str] = []
        self._centroids: Dict[str, torch.Tensor] = {}
        self._past_pairs: List[Tuple[str, str]] = []
        self._pseudo_pairs: List[Tuple[str, str]] = []
        self._pseudo_targets: List[str] = []
        self._lora: Any = None
        self._eval_wrapped = False

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        self._lora = lora
        self._install_router(model)
        name = f"{self.adapter_prefix}{int(segment.segment_id):03d}"
        spawn_segment_adapter(lora, name, freeze_others=True)
        if name not in self._adapters:
            self._adapters.append(name)
        if hasattr(lora, "set_soft_routing"):
            lora.set_soft_routing(None, None)
        return {
            "active_adapter": lora.get_active_adapter_name(),
            "num_adapters": len(self._adapters),
            "sapt_pseudo": len(self._pseudo_pairs),
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
        mix_pairs = list(pairs) + list(self._pseudo_pairs)
        mix_targets = list(targets) + list(self._pseudo_targets)
        metrics = run_ce_sft(
            model=model,
            lora=lora,
            pairs=mix_pairs,
            targets=mix_targets,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size,
            rng=self._rng,
            shuffle=True,
        )
        active = lora.get_active_adapter_name()
        self._centroids[active] = mean_embedding([p[0] for p in pairs] or [str(segment.segment_name)])
        n_arm = self._arm_synthesize(model, pairs, targets)
        self._past_pairs.extend(pairs)
        metrics["active_adapter"] = active
        metrics["sapt_arm_added"] = n_arm
        metrics["sapt_pseudo_total"] = len(self._pseudo_pairs)
        metrics["num_adapters"] = len(self._adapters)
        return metrics

    def _arm_synthesize(self, model: Any, pairs: List[Tuple[str, str]], targets: List[str]) -> int:
        if self.k <= 0 or not self._past_pairs:
            return 0
        pool = list(self._past_pairs)
        self._rng.shuffle(pool)
        take = pool[: min(self.k, len(pool))]
        prompts = [format_for_infer(model.tokenizer, ins, inp) for ins, inp in take]
        was = bool(model.model.training)
        model.model.eval()
        with torch.no_grad():
            gens = model.generate(prompts, max_new_tokens=min(32, int(getattr(model.cfg, "gen_max_new_tokens", 64))))
        if was:
            model.model.train()
        added = 0
        for pair, gen in zip(take, gens):
            text = str(gen or "").strip()
            if not text:
                continue
            self._pseudo_pairs.append(pair)
            self._pseudo_targets.append(text[:512])
            added += 1
        # cap pseudo buffer
        cap = max(self.k * max(1, len(self._adapters)), self.k)
        if len(self._pseudo_pairs) > cap:
            self._pseudo_pairs = self._pseudo_pairs[-cap:]
            self._pseudo_targets = self._pseudo_targets[-cap:]
        return added

    def _install_router(self, model: Any) -> None:
        def _route(prompts: List[str]) -> None:
            lora = self._lora
            if lora is None or not self._adapters or not hasattr(lora, "set_soft_routing"):
                return
            q = hashed_bow(prompts or [""], dim=256).mean(dim=0)
            q = F.normalize(q, dim=0, eps=1e-8)
            sims = []
            names = list(self._adapters)
            for name in names:
                c = self._centroids.get(name)
                if c is None:
                    sims.append(0.0)
                else:
                    sims.append(float(torch.dot(q, F.normalize(c, dim=0, eps=1e-8)).item()))
            t = max(1e-4, self.attn_temp)
            w = F.softmax(torch.tensor(sims, dtype=torch.float32) / t, dim=0).tolist()
            lora.set_soft_routing(names, [float(x) for x in w])
            best = names[int(max(range(len(sims)), key=lambda i: sims[i]))]
            lora.set_active_adapter(best)

        wrap_generate_router(self, model, _route)
