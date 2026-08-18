from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.data import Example, Segment
from core.formatting import format_for_infer


class RouterOnlyMethod:
    """
    Baseline D: Router-only

    - Multiple fixed/scheduled branches
    - Router selects a branch per prompt
    - No drift detector (branch setup is external/scheduled)
    """

    name = "router_only"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        router_cfg = cfg.get("router", {})
        self.num_initial_branches = int(router_cfg.get("num_initial_branches", 3))
        self._branch_names: List[str] = []

    def on_segment_start(self, *, segment: Segment, model: Any, lora: Any) -> Dict[str, Any]:
        # Ensure fixed branches exist
        if not self._branch_names:
            self._branch_names = []
            for i in range(self.num_initial_branches):
                name = f"b{i}"
                if name not in lora.list_adapters():
                    lora.create_adapter(name)
                self._branch_names.append(name)
            lora.set_active_adapter(self._branch_names[-1])
        return {"num_branches": len(self._branch_names), "active_adapter": lora.get_active_adapter_name()}

    def train_on_segment(
        self,
        *,
        segment: Segment,
        model: Any,
        lora: Any,
        router: Any,
        lora_bank: Any,
        lr: float,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """
        Router-only baseline:
          - For each training example, route to a branch and train on that branch.
          - Here we emulate routing by calling router.predict_branch and (in real impl)
            switching adapter before optimizer step.
        """

        # Initialize bank branches once (no drift logic here; fixed branches).
        if not lora_bank.list_branches():
            lora_bank.initialize(
                lora_wrapper=lora, initial_branch=self._branch_names[0], segment_id=segment.segment_id
            )
            # Create remaining branches via bank to keep registry consistent.
            for _ in self._branch_names[1:]:
                lora_bank.spawn_new_branch(lora_wrapper=lora, segment_id=segment.segment_id)
            # Switch back to latest branch by default
            lora.set_active_adapter(self._branch_names[-1])

        pairs, targets = _to_pairs(segment.train)
        metrics: Dict[str, Any] = {"batches": 0, "mean_batch_acc": 0.0, "routed_examples": 0}
        batch_accs = []

        for _ in range(max(1, epochs)):
            for b_pairs, b_targets in _batch(pairs, targets, batch_size):
                # Route each prompt in the batch and train sequentially (simplified).
                for (instruction, input_text), y in zip(b_pairs, b_targets):
                    tok = getattr(model, "tokenizer", None)
                    prompt = format_for_infer(tok, instruction, input_text) if tok is not None else f"{instruction}\n\n{input_text}"
                    decision = router.predict_branch(
                        prompt=prompt,
                        branch_names=self._branch_names,
                        branch_meta=lora_bank.state_dict(),
                        segment_id=segment.segment_id,
                    )
                    lora.set_active_adapter(decision.branch_name)
                    out = model.fit_batch([(instruction, input_text)], [y], lr=lr)
                    lora.step_adapter()
                    metrics["routed_examples"] += 1
                    batch_accs.append(float(out.get("train_batch_acc", 0.0)))
                metrics["batches"] += 1

        metrics["mean_batch_acc"] = sum(batch_accs) / max(1, len(batch_accs))
        metrics["active_adapter"] = lora.get_active_adapter_name()
        metrics["num_branches"] = len(self._branch_names)
        return metrics

    def branch_names(self) -> List[str]:
        return list(self._branch_names)


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

