from __future__ import annotations

"""SSR (ACL 2024): ICL synthesize → current-model refine → diversity filter.

Official pipeline is LLaMA-Factory scripts under `official/custom/icl_gen`.
Here: few-shot ICL generation from the frozen base (disable_adapter), refine
outputs with the current LoRA, keep K=10 diverse synthetics, mix into CE.
"""

import random
from typing import Any, Dict, List, Tuple

import torch

from core.data import Segment
from core.formatting import format_for_infer

from baselines.citb.common import run_ce_sft, to_pairs
from baselines.lit_common import cfg_seed, diversity_select, ensure_default_adapter, method_cfg, replay_k


class SSRMethod:
    name = "ssr"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        block = method_cfg(cfg, "ssr")
        self.k = replay_k(cfg, "ssr", default=10)
        self.icl_shots = int(block.get("icl_shots", 2))
        self._rng = random.Random(cfg_seed(cfg))
        self._past: List[Tuple[Tuple[str, str], str]] = []
        self._synth_pairs: List[Tuple[str, str]] = []
        self._synth_targets: List[str] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        _ = (segment, model)
        active = ensure_default_adapter(lora)
        return {
            "active_adapter": active,
            "ssr_k": self.k,
            "ssr_synth": len(self._synth_pairs),
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
        pairs, targets = to_pairs(segment.train)
        mix_pairs = list(pairs) + list(self._synth_pairs)
        mix_targets = list(targets) + list(self._synth_targets)
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
        n_new = self._synthesize(model, lora, pairs, targets)
        for p, t in zip(pairs, targets):
            self._past.append((p, t))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["ssr_added"] = n_new
        metrics["ssr_synth_total"] = len(self._synth_pairs)
        return metrics

    def _synthesize(
        self,
        model: Any,
        lora: Any,
        pairs: List[Tuple[str, str]],
        targets: List[str],
    ) -> int:
        if self.k <= 0 or not self._past:
            return 0
        shots = self._past[: max(1, self.icl_shots)]
        demo = "\n\n".join(
            f"Instruction: {sp[0]}\nInput: {sp[1]}\nOutput: {st}" for (sp, st) in shots
        )
        n_cand = min(max(self.k * 2, self.k), len(self._past))
        seeds = list(self._past)
        self._rng.shuffle(seeds)
        seeds = seeds[:n_cand]
        icl_prompts = []
        seed_pairs = []
        for (ins, inp), _old in seeds:
            block = (
                f"{demo}\n\nInstruction: {ins}\nInput: {inp}\nOutput:"
            )
            icl_prompts.append(block)
            seed_pairs.append((ins, inp))
        peft_model = getattr(lora, "peft_model", None)
        was = bool(model.model.training)
        model.model.eval()
        with torch.no_grad():
            if peft_model is not None and hasattr(peft_model, "disable_adapter"):
                with peft_model.disable_adapter():
                    drafts = model.generate(icl_prompts, max_new_tokens=32)
            else:
                drafts = model.generate(icl_prompts, max_new_tokens=32)
            refine_prompts = [
                format_for_infer(model.tokenizer, ins, inp) for ins, inp in seed_pairs
            ]
            refined = model.generate(refine_prompts, max_new_tokens=32)
        if was:
            model.model.train()
        cand_pairs: List[Tuple[str, str]] = []
        cand_tgts: List[str] = []
        for pair, draft, refn in zip(seed_pairs, drafts, refined):
            text = str(refn or draft or "").strip()
            if not text:
                continue
            cand_pairs.append(pair)
            cand_tgts.append(text[:512])
        keep_p, keep_t = diversity_select(cand_pairs, cand_tgts, self.k, self._rng)
        self._synth_pairs.extend(keep_p)
        self._synth_targets.extend(keep_t)
        cap = self.k * 8
        if len(self._synth_pairs) > cap:
            self._synth_pairs = self._synth_pairs[-cap:]
            self._synth_targets = self._synth_targets[-cap:]
        return len(keep_p)
