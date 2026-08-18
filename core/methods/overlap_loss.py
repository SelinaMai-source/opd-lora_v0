from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


def compute_overlap_loss(
    *,
    activations_by_branch: Dict[str, List[List[float]]],
    beta: float,
) -> float:
    """
    Activation diversity / anti-overlap regularization (simplified).

    Intended RP idea:
      - Encourage different LoRA branches to specialize by discouraging highly overlapping
        internal representations on the same anchor/probe set.

    Current simplified proxy:
      - Given per-branch activation vectors for the same prompts, compute average pairwise
        cosine similarity and penalize it:
            loss = beta * mean_{b1<b2} mean_i cos(a[b1,i], a[b2,i])

    Notes:
      - This is model-agnostic and works in debug mode (activations are hashed vectors).
      - In real HF model, activations can be taken from chosen layers/heads.
    """

    if beta <= 0:
        return 0.0

    branches = sorted(list(activations_by_branch.keys()))
    if len(branches) <= 1:
        return 0.0

    # Validate shapes
    n = None
    for b in branches:
        acts = activations_by_branch[b]
        if n is None:
            n = len(acts)
        elif len(acts) != n:
            raise ValueError("All branches must provide activations for the same number of prompts.")

    total = 0.0
    count = 0
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            b1, b2 = branches[i], branches[j]
            sim = _mean_cosine_similarity(activations_by_branch[b1], activations_by_branch[b2])
            total += sim
            count += 1

    mean_sim = total / max(1, count)
    return float(beta * mean_sim)


def compute_overlap_loss_torch(
    *,
    activations_by_branch: Dict[str, "Any"],  # Dict[str, torch.Tensor] shape [B, H]
    beta: float,
) -> "Any":  # torch.Tensor scalar
    """
    Differentiable anti-overlap regularization for training-time integration.

    Expects pooled, L2-normalized activations per prompt:
      activations_by_branch[b] -> Tensor[B, H]
    """
    if beta <= 0:
        import torch

        return torch.tensor(0.0, dtype=torch.float32)

    branches = sorted(list(activations_by_branch.keys()))
    if len(branches) <= 1:
        import torch

        return torch.tensor(0.0, dtype=torch.float32)

    import torch

    # Mean pairwise similarity over prompts and over branch pairs.
    total = torch.tensor(0.0, device=next(iter(activations_by_branch.values())).device)
    count = 0
    for i in range(len(branches)):
        for j in range(i + 1, len(branches)):
            b1, b2 = branches[i], branches[j]
            a1 = activations_by_branch[b1]  # [B, H]
            a2 = activations_by_branch[b2]  # [B, H]
            
            # Physics Inspiration: Electrostatic Coulomb Repulsion
            # Treat each branch's activation as a charged particle.
            # We want to minimize the electrostatic potential energy to push them apart.
            # U = 1 / (r + epsilon)
            dist = torch.norm(a1 - a2, p=2, dim=-1)
            coulomb_potential = (1.0 / (dist + 0.1)).mean()
            
            total = total + coulomb_potential
            count += 1
    mean_sim = total / max(1, count)
    return mean_sim * float(beta)


def overlap_betas_for_segment(
    cfg: Dict[str, Any],
    *,
    segment_id: int,
    num_branches: int,
) -> Dict[str, float]:
    """Curriculum schedule for anti-overlap strength.

    The regularizer is intentionally weak early on: before the bank contains
    multiple branches, overlap is not meaningful; immediately after the first
    branch split, a full-strength penalty can dominate supervised adaptation.
    """
    base_beta = float(cfg.get("beta", 0.0))
    min_branches = int(cfg.get("min_branches", 2))
    if base_beta <= 0.0 or num_branches < min_branches:
        return {
            "beta": 0.0,
            "activation_beta": 0.0,
            "weight_beta": 0.0,
            "schedule_scale": 0.0,
            "branch_scale": 0.0,
        }

    warmup_segments = max(0, int(cfg.get("warmup_segments", 1)))
    if warmup_segments <= 0:
        schedule_scale = 1.0
    else:
        schedule_scale = min(1.0, max(0.0, (float(segment_id) + 1.0) / float(warmup_segments + 1)))

    branch_scale_mode = str(cfg.get("branch_scale", "sqrt")).strip().lower()
    if branch_scale_mode == "none":
        branch_scale = 1.0
    elif branch_scale_mode == "linear":
        branch_scale = 1.0 / max(1.0, float(num_branches - 1))
    else:
        branch_scale = 1.0 / math.sqrt(max(1.0, float(num_branches - 1)))

    beta = min(float(cfg.get("max_beta", base_beta)), base_beta * schedule_scale * branch_scale)
    activation_ratio = float(cfg.get("activation_beta_ratio", 0.5))
    weight_ratio = float(cfg.get("weight_beta_ratio", 0.5))
    return {
        "beta": float(beta),
        "activation_beta": float(beta * activation_ratio),
        "weight_beta": float(beta * weight_ratio),
        "schedule_scale": float(schedule_scale),
        "branch_scale": float(branch_scale),
    }


def compute_routing_aware_orthogonal_loss(
    *,
    lora_wrapper: "Any",
    branch_probs: Dict[str, float],
    beta: float,
) -> "Any":
    """Penalize co-routed branch pairs that overlap in weight space: sum_{i<j} p_i p_j |cos(v_i,v_j)|."""
    import torch
    import torch.nn.functional as F

    if beta <= 0 or lora_wrapper is None or not hasattr(lora_wrapper, "get_adapter_vector"):
        return torch.tensor(0.0, dtype=torch.float32)

    branches = sorted(branch_probs.keys())
    if len(branches) <= 1:
        return torch.tensor(0.0, dtype=torch.float32)

    vectors: Dict[str, torch.Tensor] = {}
    for b in branches:
        try:
            vec = lora_wrapper.get_adapter_vector(b, detach=True)
        except TypeError:
            vec = lora_wrapper.get_adapter_vector(b)
        if vec.numel() > 0:
            vectors[b] = vec

    if len(vectors) <= 1:
        return torch.tensor(0.0, dtype=torch.float32)

    device = next(iter(vectors.values())).device
    total = torch.tensor(0.0, device=device)
    count = 0
    branch_list = list(vectors.keys())
    for i in range(len(branch_list)):
        for j in range(i + 1, len(branch_list)):
            b1, b2 = branch_list[i], branch_list[j]
            p_pair = float(branch_probs.get(b1, 0.0)) * float(branch_probs.get(b2, 0.0))
            if p_pair <= 0.0:
                continue
            sim = F.cosine_similarity(vectors[b1].unsqueeze(0), vectors[b2].unsqueeze(0)).squeeze()
            total = total + p_pair * sim.abs()
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=device)
    return total * float(beta)


def compute_anti_overlap_training_loss(
    *,
    activations_by_branch: Dict[str, "Any"],
    lora_wrapper: "Any",
    cfg: Dict[str, Any],
    segment_id: int,
    active_adapter: Optional[str],
) -> Tuple["Any", Dict[str, float]]:
    import torch

    betas = overlap_betas_for_segment(
        cfg,
        segment_id=segment_id,
        num_branches=len(activations_by_branch),
    )
    device = None
    if activations_by_branch:
        device = next(iter(activations_by_branch.values())).device
    total = torch.tensor(0.0, dtype=torch.float32, device=device)
    activation_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    weight_loss = torch.tensor(0.0, dtype=torch.float32, device=device)

    if betas["activation_beta"] > 0:
        activation_loss = compute_overlap_loss_torch(
            activations_by_branch=activations_by_branch,
            beta=betas["activation_beta"],
        )
        total = total + activation_loss
    if betas["weight_beta"] > 0:
        weight_loss = compute_orthogonal_weight_loss(
            lora_wrapper=lora_wrapper,
            beta=betas["weight_beta"],
            active_adapter=active_adapter,
            similarity=str(cfg.get("weight_similarity", "squared_cosine")),
        )
        if device is not None:
            weight_loss = weight_loss.to(device)
        total = total + weight_loss

    routing_beta = float(cfg.get("routing_aware_beta", 0.0))
    routing_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
    if routing_beta > 0 and lora_wrapper is not None:
        fractions = cfg.get("branch_route_fractions")
        if isinstance(fractions, dict) and fractions:
            probs = {str(k): float(v) for k, v in fractions.items()}
        else:
            branches = sorted(activations_by_branch.keys())
            probs = {b: 1.0 / len(branches) for b in branches}
        routing_loss = compute_routing_aware_orthogonal_loss(
            lora_wrapper=lora_wrapper,
            branch_probs=probs,
            beta=routing_beta,
        )
        if device is not None:
            routing_loss = routing_loss.to(device)
        total = total + routing_loss

    return total, {
        "anti_overlap_beta": float(betas["beta"]),
        "anti_overlap_activation_beta": float(betas["activation_beta"]),
        "anti_overlap_weight_beta": float(betas["weight_beta"]),
        "anti_overlap_schedule_scale": float(betas["schedule_scale"]),
        "anti_overlap_branch_scale": float(betas["branch_scale"]),
        "anti_overlap_activation_loss": float(activation_loss.detach().float().cpu().item()),
        "anti_overlap_weight_loss": float(weight_loss.detach().float().cpu().item()),
        "anti_overlap_routing_loss": float(routing_loss.detach().float().cpu().item()),
        "anti_overlap_total_loss": float(total.detach().float().cpu().item()),
    }


def compute_orthogonal_weight_loss(
    *,
    lora_wrapper: "Any",
    beta: float,
    active_adapter: Optional[str] = None,
    similarity: str = "squared_cosine",
) -> "Any":
    """
    Orthogonal regularization on LoRA weight matrices.
    Penalizes the cosine similarity between the flattened weight vectors of different branches.
    """
    import torch

    if beta <= 0 or lora_wrapper is None or not hasattr(lora_wrapper, "list_adapters"):
        return torch.tensor(0.0, dtype=torch.float32)

    branches = sorted(lora_wrapper.list_adapters())
    if len(branches) <= 1:
        return torch.tensor(0.0, dtype=torch.float32)

    vectors = {}
    for b in branches:
        # Keep gradients only for the active adapter; other branches serve as
        # fixed anchors so the regularizer does not accidentally update frozen
        # historical branches.
        detach = active_adapter is not None and b != active_adapter
        try:
            vec = lora_wrapper.get_adapter_vector(b, detach=detach)
        except TypeError:
            vec = lora_wrapper.get_adapter_vector(b)
        if vec.numel() > 0:
            vectors[b] = vec

    if len(vectors) <= 1:
        return torch.tensor(0.0, dtype=torch.float32)

    import torch.nn.functional as F

    total = torch.tensor(0.0, device=next(iter(vectors.values())).device)
    count = 0
    branch_list = list(vectors.keys())
    for i in range(len(branch_list)):
        for j in range(i + 1, len(branch_list)):
            b1, b2 = branch_list[i], branch_list[j]
            v1 = vectors[b1]
            v2 = vectors[b2]
            sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).squeeze()
            if similarity == "abs_cosine":
                penalty = sim.abs()
            elif similarity == "positive_cosine":
                penalty = F.relu(sim)
            elif similarity == "squared_cosine":
                penalty = sim.pow(2)
            elif similarity == "repulsive_energy":
                # Energy-based repulsive force: exp(sim / tau)
                tau = 0.1
                penalty = torch.exp(sim / tau)
            elif similarity == "information_bottleneck":
                # Variational Information Bottleneck (VIB)
                # Minimize mutual information between branch weight representations
                # Treat weights as samples from a Gaussian, compute KL divergence
                mu1, std1 = v1.mean(), v1.std() + 1e-6
                mu2, std2 = v2.mean(), v2.std() + 1e-6
                kl_12 = torch.log(std2/std1) + (std1.pow(2) + (mu1 - mu2).pow(2)) / (2 * std2.pow(2)) - 0.5
                kl_21 = torch.log(std1/std2) + (std2.pow(2) + (mu2 - mu1).pow(2)) / (2 * std1.pow(2)) - 0.5
                kl_sym = kl_12 + kl_21
                penalty = torch.exp(-0.1 * kl_sym)
            elif similarity == "frequency_division_multiplexing":
                # Frequency-Division Multiplexing (FDM) from Telecommunications
                # Treat the parameter vectors as temporal signals and penalize the overlap of their power spectra
                v1_fft = torch.fft.rfft(v1.float())
                v2_fft = torch.fft.rfft(v2.float())
                
                # Power spectra (magnitude squared)
                p1 = torch.abs(v1_fft).pow(2)
                p2 = torch.abs(v2_fft).pow(2)
                
                # Subsample or average pooling if the spectra are too large
                n = p1.shape[0]
                if n > 10000:
                    chunk_size = n // 1000
                    p1 = p1[:chunk_size * 1000].view(1000, chunk_size).mean(dim=1)
                    p2 = p2[:chunk_size * 1000].view(1000, chunk_size).mean(dim=1)
                
                # Normalize power spectra
                p1_norm = p1 / (p1.sum() + 1e-8)
                p2_norm = p2 / (p2.sum() + 1e-8)
                
                # Overlap penalty (dot product of power spectra)
                penalty = torch.sum(p1_norm * p2_norm)
            elif similarity == "lennard_jones":
                # Lennard-Jones Potential Repulsion (Physics-inspired)
                # V(r) = 4 * epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
                # We use dist_sq = r^2 to avoid sqrt
                epsilon = 1.0
                sigma_sq = 0.1
                dist_sq = (v1 - v2).pow(2).sum() + 1e-6
                term6 = (sigma_sq / dist_sq) ** 3
                term12 = term6 ** 2
                penalty = 4 * epsilon * (term12 - term6)
                # We only want repulsion, so if they are far enough (term6 > term12), penalty is negative.
                # We can clamp it to be >= 0 to only penalize when they are too close.
                penalty = torch.relu(penalty)
            elif similarity == "cdma":
                # Code Division Multiple Access (CDMA)
                # Assign orthogonal pseudo-random noise (PN) codes to each branch
                # Penalize the cross-correlation of their weight vectors spread by the PN codes
                import hashlib
                def get_pn_code(branch_name, size, device):
                    seed = int(hashlib.md5(branch_name.encode()).hexdigest(), 16) % (2**32)
                    gen = torch.Generator(device=device)
                    gen.manual_seed(seed)
                    return torch.sign(torch.randn(size, generator=gen, device=device))
                
                pn1 = get_pn_code(b1, v1.shape, v1.device)
                pn2 = get_pn_code(b2, v2.shape, v2.device)
                
                spread1 = v1 * pn1
                spread2 = v2 * pn2
                
                sim = F.cosine_similarity(spread1.unsqueeze(0), spread2.unsqueeze(0)).squeeze()
                penalty = sim.pow(2)
            elif similarity == "manifold_unfolding":
                # Dynamic Manifold Unfolding
                # Unfold the manifold by penalizing the RBF kernel similarity
                dist_sq = (v1 - v2).pow(2).sum()
                sigma_sq = 0.5
                penalty = torch.exp(-dist_sq / (2 * sigma_sq))
            else:
                # Default to Manifold Unfolding
                dist_sq = (v1 - v2).pow(2).sum()
                sigma_sq = 0.5
                penalty = torch.exp(-dist_sq / (2 * sigma_sq))
            total = total + penalty
            count += 1

    mean_sim = total / max(1, count)
    return mean_sim * float(beta)

def _mean_cosine_similarity(a_list: List[List[float]], b_list: List[List[float]]) -> float:
    import math

    sims = []
    for a, b in zip(a_list, b_list):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-8
        nb = math.sqrt(sum(y * y for y in b)) + 1e-8
        sims.append(dot / (na * nb))
    return float(sum(sims) / max(1, len(sims)))

