from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch

from peft import LoraConfig as PeftLoraConfig
from peft import TaskType
from peft import get_peft_model
import peft.tuners.lora.layer


# Monkey-patch PEFT LoRA Linear for soft routing and CDMA modulation
_original_lora_linear_forward = peft.tuners.lora.layer.Linear.forward

def _cdma_lora_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    soft_adapters = getattr(self, "_soft_routing_adapters", None)
    
    # We always use CDMA on top of either soft adapters or standard active adapters
    if soft_adapters is not None:
        active_weights = soft_adapters
    else:
        active_weights = [(a, 1.0) for a in getattr(self, "active_adapters", [])]

    # If no adapters are active, use base layer
    if not active_weights:
        return _original_lora_linear_forward(self, x, *args, **kwargs)
        
    result = self.base_layer(x, *args, **kwargs)
    torch_result_dtype = result.dtype
    
    lora_A_keys = self.lora_A.keys()
    
    # Cache codes to avoid regenerating on every forward pass
    if not hasattr(self, "_cdma_codes"):
        self._cdma_codes = {}
        
    def get_code(name_str, dim_size, device, dtype):
        cache_key = f"{name_str}_{dim_size}"
        if cache_key in self._cdma_codes:
            return self._cdma_codes[cache_key].to(device=device, dtype=dtype)
            
        import hashlib
        import torch
        h = int(hashlib.md5(name_str.encode('utf-8')).hexdigest(), 16)
        g = torch.Generator(device='cpu')
        g.manual_seed(h % (2**32))
        code = (torch.randint(0, 2, (dim_size,), generator=g, dtype=torch.float32) * 2 - 1)
        
        self._cdma_codes[cache_key] = code.cpu()
        return code.to(device=device, dtype=dtype)
    
    for active_adapter, weight in active_weights:
        if active_adapter not in lora_A_keys or weight == 0.0:
            continue

        lora_A = self.lora_A[active_adapter]
        lora_B = self.lora_B[active_adapter]
        dropout = self.lora_dropout[active_adapter]
        scaling = self.scaling[active_adapter]
        
        x_cast = x.to(lora_A.weight.dtype) if hasattr(self, "_cast_input_dtype") else x
        if hasattr(self, "_cast_input_dtype"):
            x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
            
        in_dim = x_cast.shape[-1]
        out_dim = result.shape[-1]
        
        # Apply CDMA encoding
        c_in = get_code(active_adapter + "_in", in_dim, x_cast.device, x_cast.dtype)
        c_out = get_code(active_adapter + "_out", out_dim, result.device, result.dtype)
        
        delta = lora_B(lora_A(dropout(x_cast * c_in))) * (scaling * weight)
        delta = delta * c_out
        
        result = result + delta
        
    return result.to(torch_result_dtype)

peft.tuners.lora.layer.Linear.forward = _cdma_lora_forward


@dataclass
class LoRAConfig:
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.0
    # PEFT accepts a list of module names or strings like "all-linear".
    target_modules: Optional[Union[str, List[str]]] = None


class LoRAWrapper:
    """
    Real PEFT LoRA wrapper with multiple adapters.

    This wrapper integrates with the repo's unified training loop:
      - `backbone.fit_batch(...)` computes loss and calls `loss.backward()`
      - `LoRAWrapper.step_adapter()` performs optimizer.step() + zero_grad()
    """

    def __init__(self, backbone: Any, cfg: LoRAConfig):
        """
        backbone: `core/models/base_model.py` backbone object.
                 Must implement:
                   - attach_peft_model(peft_model)
                   - attribute `_last_lr` updated by fit_batch(...)
        """
        self.backbone = backbone
        self.cfg = cfg
        self._active_adapter_name: str = "default"
        self._adapter_steps: Dict[str, int] = {"default": 0}
        self._frozen_adapters: Set[str] = set()

        if not self.cfg.enabled:
            # Still create a wrapper so downstream code doesn't crash, but do not add adapters.
            self.peft_model = None
            self._optimizer = None
            return

        tm = self.cfg.target_modules
        if tm is None or (isinstance(tm, list) and len(tm) == 0) or (isinstance(tm, str) and not tm.strip()):
            raise ValueError("LoRA enabled but `target_modules` is empty.")

        peft_modules: Any = tm if isinstance(tm, str) else list(tm)

        peft_cfg = PeftLoraConfig(
            r=int(self.cfg.r),
            lora_alpha=int(self.cfg.alpha),
            lora_dropout=float(self.cfg.dropout),
            bias="none",
            target_modules=peft_modules,
            task_type=TaskType.CAUSAL_LM,
        )

        # Create default adapter and attach PEFT model to backbone.
        self.peft_model = get_peft_model(self.backbone.model, peft_cfg, adapter_name="default")
        self.backbone.attach_peft_model(self.peft_model)

        # Set only the default adapter as trainable initially.
        self.set_active_adapter("default")
        self._optimizer = None
        self._rebuild_optimizer()

    def set_active_adapter(self, name: str) -> None:
        if not self.cfg.enabled:
            self._active_adapter_name = name
            return

        adapters = set(self.list_adapters())
        if name not in adapters:
            raise KeyError(f"Adapter '{name}' not found. Existing: {sorted(list(adapters))}")

        self._active_adapter_name = name
        if self.peft_model is not None:
            self.peft_model.set_adapter(name)

            # Train only the active adapter's LoRA parameters.
            # PEFT parameter names typically include: "...lora_A.<adapter_name>..." / "...lora_B.<adapter_name>..."
            for param_name, param in self.peft_model.named_parameters():
                if "lora_" in param_name:
                    param.requires_grad = (name in param_name) and (name not in self._frozen_adapters)
                else:
                    param.requires_grad = False

        # optimizer param groups already contain LoRA params; no need to rebuild here

    def save_adapter_checkpoint(self, path: str) -> None:
        """Write PEFT adapter to a directory (e.g. for overfit resume / latest snapshot)."""
        if not self.cfg.enabled or self.peft_model is None:
            return
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(str(p))

    def load_adapter_checkpoint(self, path: str) -> None:
        """Load adapter weights into the active adapter; rebuilds optimizer (Adam state reset)."""
        if not self.cfg.enabled or self.peft_model is None:
            return
        p = Path(path)
        if not p.is_dir():
            raise FileNotFoundError(f"Adapter checkpoint not found: {p}")
        self.peft_model.load_adapter(
            str(p.resolve()), adapter_name=self._active_adapter_name, is_trainable=True
        )
        self.set_active_adapter(self._active_adapter_name)
        self._rebuild_optimizer()

    def create_adapter(self, name: str) -> None:
        if not self.cfg.enabled:
            self._adapter_steps[name] = 0
            self._active_adapter_name = name
            return

        adapters = set(self.list_adapters())
        if name in adapters:
            raise KeyError(f"Adapter '{name}' already exists.")

        tm = self.cfg.target_modules
        peft_modules_add: Any = tm if isinstance(tm, str) else list(tm or [])

        peft_cfg = PeftLoraConfig(
            r=int(self.cfg.r),
            lora_alpha=int(self.cfg.alpha),
            lora_dropout=float(self.cfg.dropout),
            bias="none",
            target_modules=peft_modules_add,
            task_type=TaskType.CAUSAL_LM,
        )
        self.peft_model.add_adapter(adapter_name=name, peft_config=peft_cfg)
        self._adapter_steps[name] = 0
        self._frozen_adapters.discard(name)

        # New adapter adds new parameters, so rebuild optimizer param groups.
        self._rebuild_optimizer()

    def list_adapters(self) -> List[str]:
        if not self.cfg.enabled or self.peft_model is None:
            return list(self._adapter_steps.keys())
        # `peft_model.peft_config` is a dict: adapter_name -> config
        try:
            return list(self.peft_model.peft_config.keys())
        except Exception:
            return list(self._adapter_steps.keys())

    def trainable_parameters(self) -> List[Any]:
        if not self.cfg.enabled or self.peft_model is None:
            return []
        return [p for p in self.peft_model.parameters() if p.requires_grad]

    def freeze_adapter(self, name: str) -> None:
        if name not in self.list_adapters():
            raise KeyError(f"Adapter '{name}' not found. Existing: {sorted(self.list_adapters())}")
        self._frozen_adapters.add(name)
        if self._active_adapter_name == name:
            self.set_active_adapter(name)

    def unfreeze_adapter(self, name: str) -> None:
        self._frozen_adapters.discard(name)
        if self._active_adapter_name == name:
            self.set_active_adapter(name)

    def is_adapter_frozen(self, name: str) -> bool:
        return name in self._frozen_adapters

    def get_adapter_vector(self, name: str, *, detach: bool = True) -> torch.Tensor:
        if not self.cfg.enabled or self.peft_model is None:
            return torch.tensor([])
        tensors = []
        for param_name, param in self.peft_model.named_parameters():
            if f"lora_A.{name}." in param_name or f"lora_B.{name}." in param_name:
                tensor = param.detach() if detach else param
                tensors.append(tensor.view(-1))
        if not tensors:
            return torch.tensor([])
        return torch.cat(tensors)

    def summarize_adapter_svd(self, name: str, *, top_k: int = 8) -> Dict[str, Any]:
        """Return compact SVD triplets for one adapter's LoRA matrices."""
        if not self.cfg.enabled or self.peft_model is None:
            return {"adapter": name, "num_matrices": 0, "top_triplets": []}

        triplets: List[Dict[str, Any]] = []
        with torch.no_grad():
            for param_name, param in self.peft_model.named_parameters():
                if f"lora_A.{name}." not in param_name and f"lora_B.{name}." not in param_name:
                    continue
                matrix = param.detach().float()
                if matrix.ndim != 2 or min(matrix.shape) == 0:
                    continue
                try:
                    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                except RuntimeError:
                    continue
                limit = min(max(0, int(top_k)), int(s.numel()))
                for rank_idx in range(limit):
                    triplets.append(
                        {
                            "param_name": param_name,
                            "rank": int(rank_idx),
                            "singular_value": float(s[rank_idx].item()),
                            "left_norm": float(u[:, rank_idx].norm().item()),
                            "right_norm": float(vh[rank_idx, :].norm().item()),
                        }
                    )
        triplets.sort(key=lambda row: float(row.get("singular_value", 0.0)), reverse=True)
        return {
            "adapter": name,
            "num_matrices": int(len({row["param_name"] for row in triplets})),
            "top_triplets": triplets[: max(0, int(top_k))],
        }

    def project_active_adapter_gradients(
        self,
        *,
        reference_adapters: List[str],
        strength: float = 1.0,
        eps: float = 1e-12,
    ) -> Dict[str, Any]:
        """
        Project active adapter gradients away from previous adapter vectors.

        This is intentionally small and model-agnostic: it works on the flattened
        LoRA parameter gradient and uses previous adapter weights as subspace
        proxies. Methods can call it after backward and before `step_adapter()`.
        """
        if not self.cfg.enabled or self.peft_model is None:
            return {"hook_available": False, "projected": False, "reference_adapters": 0}
        if strength <= 0 or not reference_adapters:
            return {"hook_available": True, "projected": False, "reference_adapters": 0}

        active = self._active_adapter_name
        named_params: List[Tuple[str, Any]] = [
            (n, p)
            for n, p in self.peft_model.named_parameters()
            if (f"lora_A.{active}." in n or f"lora_B.{active}." in n) and p.requires_grad and p.grad is not None
        ]
        if not named_params:
            return {"hook_available": True, "projected": False, "reference_adapters": 0}

        grad_parts = [p.grad.detach().float().reshape(-1) for _, p in named_params]
        grad_vec = torch.cat(grad_parts)
        original_norm = float(grad_vec.norm().item())
        projected = grad_vec
        used_refs = 0
        for ref_name in reference_adapters:
            if ref_name not in self.list_adapters():
                continue
            ref_vec = self.get_adapter_vector(ref_name, detach=True).to(projected.device).float()
            if ref_vec.numel() == 0:
                continue
            width = min(int(projected.numel()), int(ref_vec.numel()))
            ref_slice = ref_vec[:width]
            denom = float(ref_slice.norm().item())
            if denom <= eps:
                continue
            unit = ref_slice / ref_slice.norm().clamp_min(eps)
            head = projected[:width]
            head = head - float(strength) * torch.dot(head, unit) * unit
            projected = torch.cat([head, projected[width:]]) if width < projected.numel() else head
            used_refs += 1

        if used_refs == 0:
            return {"hook_available": True, "projected": False, "reference_adapters": 0}

        offset = 0
        with torch.no_grad():
            for _, param in named_params:
                numel = int(param.grad.numel())
                param.grad.copy_(projected[offset : offset + numel].reshape_as(param.grad).to(param.grad.dtype))
                offset += numel

        return {
            "hook_available": True,
            "projected": True,
            "reference_adapters": int(used_refs),
            "grad_norm_before_projection": original_norm,
            "grad_norm_after_projection": float(projected.norm().item()),
        }

    def blend_adapters(self, adapters: List[str], weights: List[float], new_adapter_name: str = "blended") -> None:
        if not self.cfg.enabled or self.peft_model is None:
            return
        if new_adapter_name in self.list_adapters():
            self.peft_model.delete_adapter(new_adapter_name)
        self.peft_model.add_weighted_adapter(adapters, weights, new_adapter_name, combination_type="linear")
        self._adapter_steps[new_adapter_name] = 0
        self._frozen_adapters.add(new_adapter_name)

    def merge_adapters(self, keep_name: str, drop_name: str) -> None:
        if not self.cfg.enabled or self.peft_model is None:
            return
        
        # Average weights from drop_name into keep_name
        with torch.no_grad():
            for param_name, param in self.peft_model.named_parameters():
                if f"lora_A.{keep_name}." in param_name or f"lora_B.{keep_name}." in param_name:
                    drop_param_name = param_name.replace(f".{keep_name}.", f".{drop_name}.")
                    # Find drop param
                    drop_param = None
                    for n, p in self.peft_model.named_parameters():
                        if n == drop_param_name:
                            drop_param = p
                            break
                    if drop_param is not None:
                        param.data = (param.data + drop_param.data) / 2.0
        
        # Delete drop_name
        self.peft_model.delete_adapter(drop_name)
        if drop_name in self._adapter_steps:
            del self._adapter_steps[drop_name]
        self._frozen_adapters.discard(drop_name)
        if self._active_adapter_name == drop_name:
            self.set_active_adapter(keep_name)
        self._rebuild_optimizer()

    def step_adapter(self) -> Dict[str, Any]:
        if not self.cfg.enabled or self.peft_model is None or self._optimizer is None:
            self._adapter_steps[self._active_adapter_name] = self._adapter_steps.get(self._active_adapter_name, 0) + 1
            return {
                "grad_norm": 0.0,
                "lora_param_delta_l2": 0.0,
                "lora_params_changed": False,
                "lr": float(getattr(self.backbone, "_last_lr", 0.0) or 0.0),
            }

        # Update LR from backbone's last fit_batch(...)
        lr = float(getattr(self.backbone, "_last_lr", 0.0) or 0.0)
        if lr > 0:
            for group in self._optimizer.param_groups:
                group["lr"] = lr

        lora_named_params = [(n, p) for n, p in self.peft_model.named_parameters() if "lora_" in n and p.requires_grad]
        grad_sq_sum = 0.0
        before = []
        with torch.no_grad():
            for _, p in lora_named_params:
                if p.grad is not None:
                    grad_sq_sum += float((p.grad.detach().float() ** 2).sum().item())
                before.append(p.detach().clone())
        grad_norm = float(grad_sq_sum ** 0.5)

        self._optimizer.step()
        self._optimizer.zero_grad(set_to_none=True)

        delta_sq_sum = 0.0
        with torch.no_grad():
            for (_, p), b in zip(lora_named_params, before):
                delta = p.detach().float() - b.float()
                delta_sq_sum += float((delta ** 2).sum().item())
        delta_norm = float(delta_sq_sum ** 0.5)

        self._adapter_steps[self._active_adapter_name] = self._adapter_steps.get(self._active_adapter_name, 0) + 1
        return {
            "grad_norm": grad_norm,
            "lora_param_delta_l2": delta_norm,
            "lora_params_changed": bool(delta_norm > 0.0),
            "lr": float(lr),
        }

    def get_active_adapter_name(self) -> str:
        return self._active_adapter_name

    def set_soft_routing(self, adapters: Optional[List[str]], weights: Optional[List[float]]) -> None:
        """Enable or disable soft routing (blending at forward pass)."""
        if not self.cfg.enabled or self.peft_model is None:
            return
        soft_adapters = list(zip(adapters, weights)) if adapters and weights else None
        for module in self.peft_model.modules():
            if isinstance(module, peft.tuners.lora.layer.Linear):
                module._soft_routing_adapters = soft_adapters

    def info(self) -> Dict[str, Any]:
        total_params = 0
        trainable_params = 0
        trainable_names: List[str] = []
        if self.peft_model is not None:
            for n, p in self.peft_model.named_parameters():
                n_params = int(p.numel())
                total_params += n_params
                if p.requires_grad:
                    trainable_params += n_params
                    trainable_names.append(n)
        tm = self.cfg.target_modules
        tm_out: Any = tm if isinstance(tm, str) else (tm or [])
        return {
            "enabled": self.cfg.enabled,
            "r": self.cfg.r,
            "alpha": self.cfg.alpha,
            "dropout": self.cfg.dropout,
            "target_modules": tm_out,
            "active_adapter": self._active_adapter_name,
            "adapters": dict(self._adapter_steps),
            "frozen_adapters": sorted(self._frozen_adapters),
            "total_parameters": int(total_params),
            "trainable_parameters": int(trainable_params),
            "trainable_parameter_names": trainable_names,
        }

    def _rebuild_optimizer(self) -> None:
        if not self.cfg.enabled or self.peft_model is None:
            self._optimizer = None
            return

        # Include all LoRA parameters of all adapters currently registered.
        lora_params = [p for n, p in self.peft_model.named_parameters() if "lora_" in n]
        if not lora_params:
            self._optimizer = None
            return

        # AdamW is a common default; training hyperparameters are controlled via `lr` passed to fit_batch.
        self._optimizer = torch.optim.AdamW(lora_params, lr=1e-4)


class DebugLoRAWrapper:
    """
    Debug/no-op LoRA wrapper used when the backbone is the repo's DebugTextModel.

    It keeps the adapter-selection API stable so the rest of the pipeline can run.
    """

    def __init__(self, cfg: LoRAConfig):
        self.cfg = cfg
        self._active_adapter_name: str = "default"
        self._adapter_steps: Dict[str, int] = {"default": 0}
        self._frozen_adapters: Set[str] = set()
        self._adapter_vectors: Dict[str, torch.Tensor] = {"default": self._make_debug_vector("default")}

    def set_active_adapter(self, name: str) -> None:
        if name not in self._adapter_steps:
            raise KeyError(f"Adapter '{name}' not found. Existing: {sorted(list(self._adapter_steps.keys()))}")
        self._active_adapter_name = name

    def create_adapter(self, name: str) -> None:
        if name in self._adapter_steps:
            raise KeyError(f"Adapter '{name}' already exists.")
        self._adapter_steps[name] = 0
        self._frozen_adapters.discard(name)
        self._adapter_vectors[name] = self._make_debug_vector(name)

    def list_adapters(self) -> List[str]:
        return list(self._adapter_steps.keys())

    def trainable_parameters(self) -> List[Any]:
        return []

    def freeze_adapter(self, name: str) -> None:
        if name not in self._adapter_steps:
            raise KeyError(f"Adapter '{name}' not found. Existing: {sorted(list(self._adapter_steps.keys()))}")
        self._frozen_adapters.add(name)

    def unfreeze_adapter(self, name: str) -> None:
        self._frozen_adapters.discard(name)

    def is_adapter_frozen(self, name: str) -> bool:
        return name in self._frozen_adapters

    def step_adapter(self) -> Dict[str, Any]:
        self._adapter_steps[self._active_adapter_name] = self._adapter_steps.get(self._active_adapter_name, 0) + 1
        if self._active_adapter_name not in self._frozen_adapters:
            step = float(self._adapter_steps[self._active_adapter_name])
            self._adapter_vectors[self._active_adapter_name] = (
                self._adapter_vectors.get(self._active_adapter_name, self._make_debug_vector(self._active_adapter_name))
                + 0.001 * step
            )
        return {
            "grad_norm": 0.0,
            "lora_param_delta_l2": 0.0,
            "lora_params_changed": False,
            "lr": 0.0,
        }

    def get_active_adapter_name(self) -> str:
        return self._active_adapter_name

    def set_soft_routing(self, adapters: Optional[List[str]], weights: Optional[List[float]]) -> None:
        pass

    def get_adapter_vector(self, name: str, *, detach: bool = True) -> torch.Tensor:
        vec = self._adapter_vectors.get(name, self._make_debug_vector(name))
        return vec.detach().clone() if detach else vec.clone()

    def summarize_adapter_svd(self, name: str, *, top_k: int = 8) -> Dict[str, Any]:
        vec = self.get_adapter_vector(name, detach=True).float()
        if vec.numel() == 0:
            return {"adapter": name, "num_matrices": 0, "top_triplets": []}
        side = int(max(1, vec.numel() ** 0.5))
        matrix = vec[: side * side].reshape(side, side)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        triplets = []
        for rank_idx in range(min(max(0, int(top_k)), int(s.numel()))):
            triplets.append(
                {
                    "param_name": f"debug_lora_vector.{name}",
                    "rank": int(rank_idx),
                    "singular_value": float(s[rank_idx].item()),
                    "left_norm": float(u[:, rank_idx].norm().item()),
                    "right_norm": float(vh[rank_idx, :].norm().item()),
                }
            )
        return {"adapter": name, "num_matrices": 1, "top_triplets": triplets}

    def project_active_adapter_gradients(
        self,
        *,
        reference_adapters: List[str],
        strength: float = 1.0,
        eps: float = 1e-12,
    ) -> Dict[str, Any]:
        _ = eps
        return {
            "hook_available": True,
            "projected": False,
            "reference_adapters": int(len(reference_adapters)),
            "projection_strength": float(strength),
            "debug_noop": True,
        }

    def save_adapter_checkpoint(self, path: str) -> None:
        return

    def load_adapter_checkpoint(self, path: str) -> None:
        return

    def _make_debug_vector(self, name: str) -> torch.Tensor:
        import hashlib

        seed = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % (2**32)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        return torch.randn(16, generator=gen, dtype=torch.float32)

    def info(self) -> Dict[str, Any]:
        tm = self.cfg.target_modules
        tm_out: Any = tm if isinstance(tm, str) else (tm or [])
        return {
            "enabled": self.cfg.enabled,
            "r": self.cfg.r,
            "alpha": self.cfg.alpha,
            "dropout": self.cfg.dropout,
            "target_modules": tm_out,
            "active_adapter": self._active_adapter_name,
            "adapters": dict(self._adapter_steps),
            "frozen_adapters": sorted(self._frozen_adapters),
        }


def build_lora_wrapper(base_model: Any, lora_cfg_dict: Dict[str, Any]) -> LoRAWrapper:
    raw_tm = lora_cfg_dict.get("target_modules", [])
    if isinstance(raw_tm, str):
        parsed_tm: Optional[Union[str, List[str]]] = raw_tm.strip()
    else:
        lst = list(raw_tm or [])
        parsed_tm = lst if lst else None

    cfg = LoRAConfig(
        enabled=bool(lora_cfg_dict.get("enabled", True)),
        r=int(lora_cfg_dict.get("r", 16)),
        alpha=int(lora_cfg_dict.get("alpha", 32)),
        dropout=float(lora_cfg_dict.get("dropout", 0.0)),
        target_modules=parsed_tm,
    )
    # Debug backbones (DebugTextModel) don't have PEFT hooks.
    if not hasattr(base_model, "attach_peft_model") or not hasattr(base_model, "model"):
        return DebugLoRAWrapper(cfg=cfg)  # type: ignore[return-value]

    return LoRAWrapper(backbone=base_model, cfg=cfg)

