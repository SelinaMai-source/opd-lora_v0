from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class BranchInfo:
    name: str
    created_at_segment: int
    frozen: bool = False


class LoRABank:
    """
    A lightweight LoRA bank manager.

    Responsibilities:
      - Track branches (adapters) and the active branch
      - Spawn new branches and optionally freeze old ones
      - Enforce max_branches policy (simplified)

    This manager is intentionally model-agnostic. It delegates adapter creation/selection
    to the passed LoRAWrapper.
    """

    def __init__(self, *, max_branches: int = 32):
        self.max_branches = int(max_branches)
        self._branches: Dict[str, BranchInfo] = {}
        self._active: Optional[str] = None
        self._lora_wrapper: Optional[Any] = None

    def initialize(self, *, lora_wrapper: Any, initial_branch: str = "b0", segment_id: int = 0) -> None:
        self._lora_wrapper = lora_wrapper
        if initial_branch not in lora_wrapper.list_adapters():
            lora_wrapper.create_adapter(initial_branch)
        lora_wrapper.set_active_adapter(initial_branch)

        self._branches[initial_branch] = BranchInfo(
            name=initial_branch, created_at_segment=segment_id, frozen=False
        )
        self._active = initial_branch

    def get_active_branch(self) -> str:
        if self._active is None:
            raise RuntimeError("LoRABank not initialized: active branch is None")
        return self._active

    def list_branches(self) -> List[str]:
        return list(self._branches.keys())

    def list_frozen_branches(self) -> List[str]:
        return [name for name, info in self._branches.items() if bool(info.frozen)]

    def list_trainable_branches(self) -> List[str]:
        return [name for name, info in self._branches.items() if not bool(info.frozen)]

    def is_branch_frozen(self, name: str) -> bool:
        info = self._branches.get(name)
        if info is None:
            raise KeyError(f"Branch '{name}' not found. Existing: {sorted(self._branches)}")
        return bool(info.frozen)

    def freeze_current_branch(self) -> None:
        b = self.get_active_branch()
        self._branches[b].frozen = True
        if self._lora_wrapper is not None and hasattr(self._lora_wrapper, "freeze_adapter"):
            self._lora_wrapper.freeze_adapter(b)

    def merge_most_similar_branches(self) -> None:
        """
        Find the two most similar frozen branches by cosine similarity of their weights,
        and merge them using weight averaging to free up capacity.
        """
        if self._lora_wrapper is None or not hasattr(self._lora_wrapper, "get_adapter_vector"):
            return
        
        frozen = self.list_frozen_branches()
        if len(frozen) < 2:
            return
            
        import torch
        import torch.nn.functional as F
        
        vectors = {}
        for b in frozen:
            vec = self._lora_wrapper.get_adapter_vector(b)
            if vec.numel() > 0:
                vectors[b] = vec
                
        if len(vectors) < 2:
            return
            
        best_sim = -float("inf")
        best_pair = (None, None)
        
        frozen_list = list(vectors.keys())
        for i in range(len(frozen_list)):
            for j in range(i + 1, len(frozen_list)):
                b1, b2 = frozen_list[i], frozen_list[j]
                sim = F.cosine_similarity(vectors[b1].unsqueeze(0), vectors[b2].unsqueeze(0)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (b1, b2)
                    
        keep_name, drop_name = best_pair
        if keep_name and drop_name:
            # Merge drop_name into keep_name
            if hasattr(self._lora_wrapper, "merge_adapters"):
                self._lora_wrapper.merge_adapters(keep_name, drop_name)
            del self._branches[drop_name]

    def spawn_new_branch(self, *, lora_wrapper: Any, segment_id: int) -> str:
        """
        Create a new branch and switch to it.
        If max_branches reached, merge the two most similar frozen branches to free capacity.
        """

        self._lora_wrapper = lora_wrapper
        if len(self._branches) >= self.max_branches:
            self.merge_most_similar_branches()
            
        # Find an available name
        for i in range(self.max_branches + 100):
            name = f"b{i}"
            if name not in self._branches:
                break
        else:
            name = f"b{len(self._branches)}_overflow_s{segment_id}"

        if name not in lora_wrapper.list_adapters():
            lora_wrapper.create_adapter(name)
        lora_wrapper.set_active_adapter(name)

        self._branches[name] = BranchInfo(name=name, created_at_segment=segment_id, frozen=False)
        self._active = name
        return name

    def blend_adapters(self, adapters: List[str], weights: List[float], new_adapter_name: str = "blended") -> None:
        if self._lora_wrapper is not None and hasattr(self._lora_wrapper, "blend_adapters"):
            self._lora_wrapper.blend_adapters(adapters, weights, new_adapter_name)

    def set_soft_routing(self, adapters: Optional[List[str]], weights: Optional[List[float]]) -> None:
        if self._lora_wrapper is not None and hasattr(self._lora_wrapper, "set_soft_routing"):
            self._lora_wrapper.set_soft_routing(adapters, weights)

    def set_active_adapter(self, name: str) -> None:
        """
        Switch the underlying PEFT adapter if the bank is backed by a LoRAWrapper.
        This is needed for evaluation-time routing where we may choose a branch
        per prompt and then generate with the selected adapter.
        """
        if self._lora_wrapper is None:
            raise RuntimeError("LoRABank has no lora_wrapper reference; initialize/spawn must be called first.")
        self._lora_wrapper.set_active_adapter(name)

    def available_adapters(self) -> List[str]:
        if self._lora_wrapper is None or not hasattr(self._lora_wrapper, "list_adapters"):
            return self.list_branches()
        return list(self._lora_wrapper.list_adapters())

    def has_adapter(self, name: str) -> bool:
        return name in set(self.available_adapters())

    def state_dict(self) -> Dict[str, Any]:
        return {
            "max_branches": self.max_branches,
            "active": self._active,
            "trainable_branches": self.list_trainable_branches(),
            "frozen_branches": self.list_frozen_branches(),
            "branches": {k: vars(v) for k, v in self._branches.items()},
        }

