from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from core.causal_lm_metrics import count_supervised_label_tokens, teacher_forced_token_accuracy_shifted
from core.data import Example, Segment
from core.formatting import format_for_infer, format_for_train
from core.train_labels import build_supervised_labels
from core.metrics_utils import (
    dialogue_slot_error_rate as _dialogue_slot_error_rate,
    lcs_length as _lcs_length,
    lcs_overlap as _lcs_overlap,
    rouge_l_fscore as _rouge_l_fscore,
    sentence_bleu4 as _sentence_bleu4,
    starts_incorrectly as _starts_incorrectly,
    token_f1 as _token_f1,
)
from core.normalize_answer import basic_answer_normalize as _basic_answer_normalize
from core.normalize_answer import normalize_for_eval as _normalize


@dataclass
class EvalResult:
    current_score: float
    seen_avg_score: float
    current_task_aware_score: float
    seen_avg_task_aware_score: float
    forgetting: float
    task_aware_forgetting: float
    num_seen_segments: int
    token_f1_mean: float
    lcs_overlap_mean: float
    rouge_l_mean: float
    bleu_mean: float
    slot_error_rate: float
    extra: Dict[str, Any]


def _extract_router_feature_for_prompt(model: Any, router: Any, lora_bank: Any, prompt: str) -> Any:
    import torch

    feature_adapter = (
        router.feature_adapter_name
        if hasattr(lora_bank, "has_adapter") and lora_bank.has_adapter(router.feature_adapter_name)
        else None
    )
    active_before = lora_bank.get_active_branch() if hasattr(lora_bank, "get_active_branch") else None
    if feature_adapter is not None and hasattr(lora_bank, "set_active_adapter"):
        lora_bank.set_active_adapter(feature_adapter)
    try:
        if hasattr(model, "get_activations_tensor"):
            return model.get_activations_tensor([prompt], with_grad=False)
        return torch.tensor(model.get_activations([prompt]), dtype=torch.float32)
    finally:
        if active_before is not None and hasattr(lora_bank, "set_active_adapter") and lora_bank.has_adapter(active_before):
            lora_bank.set_active_adapter(active_before)


def _normalized_routing_scores(scores: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, float(v)) for v in scores.values()))
    if total <= 0:
        n = max(1, len(scores))
        return {k: 1.0 / n for k in scores}
    return {k: float(max(0.0, float(v)) / total) for k, v in scores.items()}


def _routing_entropy(prob_scores: Dict[str, float]) -> float:
    import math

    entropy = 0.0
    for p in prob_scores.values():
        if p > 0:
            entropy -= float(p) * math.log(float(p) + 1e-12)
    return float(entropy)


def _orthogonal_gate_blend_weights(
    adapters: List[str],
    weights: List[float],
    lora_bank: Any,
    lam: float,
) -> List[float]:
    """Down-weight blend coefficients when candidate LoRA branches overlap in weight space.

    w_i' ∝ w_i · exp(-λ Σ_{j≠i} |cos(v_i, v_j)|), then renormalized.
    Eval-time only; uses detached adapter vectors (no label leakage).
    """
    import math

    import torch
    import torch.nn.functional as F

    if lam <= 0.0 or len(adapters) <= 1:
        return weights

    wrapper = getattr(lora_bank, "_lora_wrapper", None)
    if wrapper is None or not hasattr(wrapper, "get_adapter_vector"):
        return weights

    vecs: Dict[str, torch.Tensor] = {}
    for name in adapters:
        try:
            vec = wrapper.get_adapter_vector(name, detach=True)
        except TypeError:
            vec = wrapper.get_adapter_vector(name)
        if vec.numel() > 0:
            vecs[name] = F.normalize(vec.float(), dim=0)

    if len(vecs) < 2:
        return weights

    gated: List[float] = []
    for i, name in enumerate(adapters):
        w = float(weights[i])
        if name not in vecs:
            gated.append(w)
            continue
        sim_sum = 0.0
        for j, other in enumerate(adapters):
            if i == j or other not in vecs:
                continue
            sim_sum += abs(
                float(F.cosine_similarity(vecs[name].unsqueeze(0), vecs[other].unsqueeze(0)).item())
            )
        gated.append(w * math.exp(-lam * sim_sum))

    total = float(sum(gated))
    if total <= 0.0:
        return weights
    return [g / total for g in gated]


def _oracle_branch_for_example(model: Any, lora_bank: Any, ex: Example, branch_names: List[str]) -> Dict[str, Any]:
    pairs = [(ex.instruction, ex.input)]
    targets = [ex.output]
    active_before = lora_bank.get_active_branch() if hasattr(lora_bank, "get_active_branch") else None
    losses: Dict[str, float] = {}
    try:
        for branch_name in branch_names:
            if hasattr(lora_bank, "set_active_adapter"):
                lora_bank.set_active_adapter(branch_name)
            losses[branch_name] = float(model.score_answer_nlls(pairs, targets)[0])
    finally:
        if active_before is not None and hasattr(lora_bank, "set_active_adapter") and lora_bank.has_adapter(active_before):
            lora_bank.set_active_adapter(active_before)

    ranked = sorted((loss, name) for name, loss in losses.items())
    best_loss, best_branch = ranked[0]
    second_loss = ranked[1][0] if len(ranked) > 1 else best_loss
    return {
        "oracle_branch": best_branch,
        "oracle_best_loss": float(best_loss),
        "oracle_margin": float(second_loss - best_loss),
        "oracle_losses": losses,
    }


def evaluate_stream(
    *,
    model: Any,
    segments_seen: List[Segment],
    max_new_tokens: int = 64,
    router: Optional[Any] = None,
    lora_bank: Optional[Any] = None,
    segment_id: int,
    normalization_cfg: Optional[Dict[str, Any]] = None,
    save_debug_examples_dir: Optional[str] = None,
    historical_best_per_segment: Optional[Dict[int, float]] = None,
    historical_best_task_aware_per_segment: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    """
    Unified evaluation for continual instruction tuning.

    Returns a metrics dict that is stable across modes:
      - current_score: accuracy on current segment eval
      - seen_avg_score: average accuracy across all seen segments
      - forgetting: max(prev_best - current) over seen segments (simplified)
      - num_seen_segments

    Extension points:
      - drift metrics
      - routing metrics (branch usage entropy, decision confidence)
      - overlap metrics (activation similarity)
    """

    # Track per-segment accuracy over time
    per_seg_acc: List[Tuple[int, float]] = []
    per_seg_task_aware_acc: List[Tuple[int, float]] = []
    routing_stats = {
        "num_routed": 0,
        "branch_counts": {},
        "oracle_best_branch_counts": {},
        "oracle_agreement_count": 0,
        "oracle_num_examples": 0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "oracle_margin_sum": 0.0,
    }

    all_examples_for_dump: List[Dict[str, Any]] = []
    all_token_f1: List[float] = []
    all_lcs_overlap: List[float] = []
    all_rouge_l: List[float] = []
    all_bleu: List[float] = []
    all_slot_error: List[float] = []
    all_task_aware_scores: List[float] = []
    task_score_type_counts: Dict[str, int] = {}
    all_prefix1: List[int] = []
    all_prefix3: List[int] = []
    all_prefix5: List[int] = []
    num_bad_prefix = 0
    for seg in segments_seen:
        acc, task_aware_acc, seg_routing, seg_examples = _eval_segment(
            model=model,
            segment=seg,
            max_new_tokens=max_new_tokens,
            router=router,
            lora_bank=lora_bank,
            segment_id=segment_id,
            normalization_cfg=normalization_cfg or {},
        )
        per_seg_acc.append((seg.segment_id, acc))
        per_seg_task_aware_acc.append((seg.segment_id, task_aware_acc))
        _merge_routing_stats(routing_stats, seg_routing)
        all_examples_for_dump.extend(seg_examples)
        all_token_f1.extend([float(x.get("token_f1", 0.0)) for x in seg_examples])
        all_lcs_overlap.extend([float(x.get("lcs_overlap", 0.0)) for x in seg_examples])
        all_rouge_l.extend([float(x.get("rouge_l", 0.0)) for x in seg_examples])
        all_bleu.extend([float(x.get("bleu", 0.0)) for x in seg_examples])
        all_slot_error.extend(
            [float(x["slot_error_rate"]) for x in seg_examples if x.get("slot_error_rate") is not None]
        )
        all_task_aware_scores.extend([float(x.get("task_aware_score", 0.0)) for x in seg_examples])
        for x in seg_examples:
            score_type = str(x.get("task_score_type", "unknown"))
            task_score_type_counts[score_type] = task_score_type_counts.get(score_type, 0) + 1
        num_bad_prefix += sum(1 for x in seg_examples if bool(x.get("bad_prefix_mismatch", False)))
        all_prefix1.extend([int(bool(x.get("prefix_1_match", False))) for x in seg_examples])
        all_prefix3.extend([int(bool(x.get("prefix_3_match", False))) for x in seg_examples])
        all_prefix5.extend([int(bool(x.get("prefix_5_match", False))) for x in seg_examples])

    # current segment is the last in segments_seen
    current_score = per_seg_acc[-1][1] if per_seg_acc else 0.0
    seen_avg_score = sum(a for _, a in per_seg_acc) / max(1, len(per_seg_acc))
    current_task_aware_score = per_seg_task_aware_acc[-1][1] if per_seg_task_aware_acc else 0.0
    seen_avg_task_aware_score = sum(a for _, a in per_seg_task_aware_acc) / max(1, len(per_seg_task_aware_acc))
    anytime_score = float(seen_avg_score)
    anytime_task_aware_score = float(seen_avg_task_aware_score)

    if historical_best_per_segment is None:
        historical_best_per_segment = {}
    previous_segment_ids = {sid for sid, _ in per_seg_acc[:-1]}
    forgetting_by_segment: List[Dict[str, float]] = []
    forgetting_values: List[float] = []
    for sid, acc in per_seg_acc:
        best_before = float(historical_best_per_segment.get(sid, acc))
        seg_forgetting = float(max(0.0, best_before - acc)) if sid in previous_segment_ids else 0.0
        if sid in previous_segment_ids:
            forgetting_values.append(seg_forgetting)
        forgetting_by_segment.append(
            {
                "segment_id": int(sid),
                "accuracy": float(acc),
                "best_historical_accuracy": float(best_before),
                "forgetting": float(seg_forgetting),
            }
        )
        historical_best_per_segment[sid] = max(best_before, float(acc))
    forgetting = float(sum(forgetting_values) / max(1, len(forgetting_values))) if forgetting_values else 0.0

    if historical_best_task_aware_per_segment is None:
        historical_best_task_aware_per_segment = {}
    previous_task_segment_ids = {sid for sid, _ in per_seg_task_aware_acc[:-1]}
    task_aware_forgetting_by_segment: List[Dict[str, float]] = []
    task_aware_forgetting_values: List[float] = []
    for sid, acc in per_seg_task_aware_acc:
        best_before = float(historical_best_task_aware_per_segment.get(sid, acc))
        seg_forgetting = float(max(0.0, best_before - acc)) if sid in previous_task_segment_ids else 0.0
        if sid in previous_task_segment_ids:
            task_aware_forgetting_values.append(seg_forgetting)
        task_aware_forgetting_by_segment.append(
            {
                "segment_id": int(sid),
                "task_aware_accuracy": float(acc),
                "best_historical_task_aware_accuracy": float(best_before),
                "task_aware_forgetting": float(seg_forgetting),
            }
        )
        historical_best_task_aware_per_segment[sid] = max(best_before, float(acc))
    task_aware_forgetting = (
        float(sum(task_aware_forgetting_values) / max(1, len(task_aware_forgetting_values)))
        if task_aware_forgetting_values
        else 0.0
    )

    routing_num = int(routing_stats.get("num_routed", 0))
    branch_counts = dict(routing_stats.get("branch_counts", {}))
    branch_utilization = {
        branch: float(count) / max(1, routing_num)
        for branch, count in sorted(branch_counts.items())
    }
    oracle_num = int(routing_stats.get("oracle_num_examples", 0))

    extra = {
        "per_segment_accuracy": [{"segment_id": sid, "accuracy": acc} for sid, acc in per_seg_acc],
        "per_segment_task_aware_accuracy": [
            {"segment_id": sid, "task_aware_accuracy": acc} for sid, acc in per_seg_task_aware_acc
        ],
        "anytime_score": anytime_score,
        "anytime_task_aware_score": anytime_task_aware_score,
        "forgetting_by_segment": forgetting_by_segment,
        "task_aware_forgetting_by_segment": task_aware_forgetting_by_segment,
        "routing": {
            **routing_stats,
            "branch_utilization": branch_utilization,
            "decision_confidence_mean": float(routing_stats.get("confidence_sum", 0.0) / max(1, routing_num)),
            "decision_entropy_mean": float(routing_stats.get("entropy_sum", 0.0) / max(1, routing_num)),
            "oracle_agreement_rate": float(routing_stats.get("oracle_agreement_count", 0) / max(1, oracle_num)),
            "oracle_margin_mean": float(routing_stats.get("oracle_margin_sum", 0.0) / max(1, oracle_num)),
        },
        "token_f1_mean": float(sum(all_token_f1) / max(1, len(all_token_f1))),
        "lcs_overlap_mean": float(sum(all_lcs_overlap) / max(1, len(all_lcs_overlap))),
        "rouge_l_mean": float(sum(all_rouge_l) / max(1, len(all_rouge_l))),
        "bleu_mean": float(sum(all_bleu) / max(1, len(all_bleu))),
        "slot_error_rate": float(sum(all_slot_error) / max(1, len(all_slot_error))),
        "slot_error_count": int(len(all_slot_error)),
        "task_aware_score_mean": float(sum(all_task_aware_scores) / max(1, len(all_task_aware_scores))),
        "task_score_type_counts": task_score_type_counts,
        "prefix_1_match_mean": float(sum(all_prefix1) / max(1, len(all_prefix1))),
        "prefix_3_match_mean": float(sum(all_prefix3) / max(1, len(all_prefix3))),
        "prefix_5_match_mean": float(sum(all_prefix5) / max(1, len(all_prefix5))),
        "num_bad_prefix_mismatch": int(num_bad_prefix),
    }
    extra["likely_assistant_start_or_continuation_boundary"] = bool(
        extra["prefix_1_match_mean"] < 0.5 if (len(all_prefix1) > 0) else False
    )
    if save_debug_examples_dir:
        tok = getattr(model, "tokenizer", None)
        _save_debug_examples(
            save_dir=save_debug_examples_dir,
            segment_id=segment_id,
            examples=all_examples_for_dump[: max(5, min(50, len(all_examples_for_dump)))],
            normalization_cfg=normalization_cfg or {},
            generation_cfg={
                "requested_max_new_tokens": max_new_tokens,
                "effective_max_new_tokens": _resolve_eval_max_new_tokens(
                    max_new_tokens=max_new_tokens,
                    eval_examples=[ex for seg in segments_seen for ex in seg.eval],
                    normalization_cfg=normalization_cfg or {},
                ),
                "do_sample": False,
                "eos_token_id": getattr(tok, "eos_token_id", None),
                "pad_token_id": getattr(tok, "pad_token_id", None),
            },
        )
    return asdict(
        EvalResult(
            current_score=float(current_score),
            seen_avg_score=float(seen_avg_score),
            current_task_aware_score=float(current_task_aware_score),
            seen_avg_task_aware_score=float(seen_avg_task_aware_score),
            forgetting=float(forgetting),
            task_aware_forgetting=float(task_aware_forgetting),
            num_seen_segments=int(len(segments_seen)),
            token_f1_mean=float(sum(all_token_f1) / max(1, len(all_token_f1))),
            lcs_overlap_mean=float(sum(all_lcs_overlap) / max(1, len(all_lcs_overlap))),
            rouge_l_mean=float(sum(all_rouge_l) / max(1, len(all_rouge_l))),
            bleu_mean=float(sum(all_bleu) / max(1, len(all_bleu))),
            slot_error_rate=float(sum(all_slot_error) / max(1, len(all_slot_error))),
            extra=extra,
        )
    )


def _eval_segment(
    *,
    model: Any,
    segment: Segment,
    max_new_tokens: int,
    router: Optional[Any],
    lora_bank: Optional[Any],
    segment_id: int,
    normalization_cfg: Dict[str, Any],
) -> Tuple[float, float, Dict[str, Any], List[Dict[str, Any]]]:
    correct = 0
    task_aware_correct = 0
    total = 0

    routing_stats = {
        "num_routed": 0,
        "branch_counts": {},
        "oracle_best_branch_counts": {},
        "oracle_agreement_count": 0,
        "oracle_num_examples": 0,
        "confidence_sum": 0.0,
        "entropy_sum": 0.0,
        "oracle_margin_sum": 0.0,
    }

    tok = getattr(model, "tokenizer", None)
    if tok is None:
        raise RuntimeError("Model has no tokenizer; cannot format chat prompts for evaluation.")
    eval_examples = list(segment.eval)
    enable_infer_token_audit = bool(normalization_cfg.get("enable_infer_token_audit", False))
    enable_teacher_forced_eval = bool(normalization_cfg.get("enable_teacher_forced_eval", False))
    infer_token_audit_max_examples = int(normalization_cfg.get("infer_token_audit_max_examples", 8))
    teacher_forced_eval_max_examples = int(normalization_cfg.get("teacher_forced_eval_max_examples", 8))

    prompts = [format_for_infer(tok, ex.instruction, ex.input, add_generation_prompt=True) for ex in eval_examples]
    targets = [ex.output for ex in eval_examples]
    effective_max_new_tokens = _resolve_eval_max_new_tokens(
        max_new_tokens=max_new_tokens,
        eval_examples=eval_examples,
        normalization_cfg=normalization_cfg,
    )

    # If router/bank available, route per prompt (simplified hard routing).
    audited = enable_infer_token_audit and router is None and lora_bank is None and hasattr(model, "generate_with_ids")

    if router is not None and lora_bank is not None:
        branch_names = lora_bank.list_branches()
        branch_meta = lora_bank.state_dict()
        preds: List[str] = []
        routing_details: List[Dict[str, Any]] = []
        for ex, p in zip(eval_examples, prompts):
            features = _extract_router_feature_for_prompt(model, router, lora_bank, p)
            decision = router.predict_branch(
                prompt=p,
                branch_names=branch_names,
                branch_meta=branch_meta,
                segment_id=segment_id,
                features=features,
            )
            prob_scores = _normalized_routing_scores(decision.scores)
            use_nll_arbitration = bool(getattr(router, "nll_arbitration", False))
            ranked = sorted(prob_scores.items(), key=lambda kv: kv[1], reverse=True) if prob_scores else []
            margin = float(ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else 1.0
            arbitrated_low_margin = False
            # Verify-then-route: label-free prompt-NLL arbitration on low-margin
            # decisions. Works with soft routing too — uncertain cases hard-route
            # to the NLL winner instead of destructive parameter blending.
            if (
                use_nll_arbitration
                and hasattr(model, "score_prompt_nlls")
                and hasattr(lora_bank, "set_active_adapter")
                and len(prob_scores) > 1
                and margin < float(getattr(router, "arbitration_margin", 0.15))
            ):
                top_k = max(2, int(getattr(router, "arbitration_top_k", 3)))
                candidates = [b for b, _ in ranked[:top_k]]
                cand_nlls: Dict[str, float] = {}
                for cand in candidates:
                    lora_bank.set_active_adapter(cand)
                    cand_nlls[cand] = float(model.score_prompt_nlls([p])[0])
                arbitrated = min(cand_nlls, key=cand_nlls.get)
                routing_stats["nll_arbitration_count"] = routing_stats.get("nll_arbitration_count", 0) + 1
                if arbitrated != decision.branch_name:
                    routing_stats["nll_arbitration_changed"] = routing_stats.get("nll_arbitration_changed", 0) + 1
                decision.branch_name = arbitrated
                decision.reason = f"{decision.reason}+nll_arbitration"
                arbitrated_low_margin = True
                prob_scores = {b: (1.0 if b == arbitrated else 0.0) for b in prob_scores}
            routing_stats["num_routed"] += 1
            routing_stats["branch_counts"][decision.branch_name] = (
                routing_stats["branch_counts"].get(decision.branch_name, 0) + 1
            )
            routing_stats["confidence_sum"] += float(max(prob_scores.values()) if prob_scores else 0.0)
            routing_stats["entropy_sum"] += float(_routing_entropy(prob_scores))
            oracle = _oracle_branch_for_example(model, lora_bank, ex, branch_names)
            routing_stats["oracle_num_examples"] += 1
            routing_stats["oracle_agreement_count"] += int(decision.branch_name == oracle["oracle_branch"])
            routing_stats["oracle_margin_sum"] += float(oracle["oracle_margin"])
            routing_stats["oracle_best_branch_counts"][oracle["oracle_branch"]] = (
                routing_stats["oracle_best_branch_counts"].get(oracle["oracle_branch"], 0) + 1
            )
            routing_details.append(
                {
                    "routing_selected_branch": decision.branch_name,
                    "routing_confidence": float(max(prob_scores.values()) if prob_scores else 0.0),
                    "routing_entropy": float(_routing_entropy(prob_scores)),
                    "routing_oracle_branch": oracle["oracle_branch"],
                    "routing_oracle_margin": float(oracle["oracle_margin"]),
                    "routing_oracle_best_loss": float(oracle["oracle_best_loss"]),
                }
            )
            margin_gate_hard = bool(
                getattr(router, "margin_gated_soft_routing", False)
                and margin >= float(getattr(router, "margin_gate_threshold", 0.12))
            )
            if margin_gate_hard:
                routing_stats["margin_gate_hard_count"] = routing_stats.get("margin_gate_hard_count", 0) + 1
            # Switch adapter before generating (needed for real multi-adapter evaluation).
            if (
                getattr(router, "soft_routing", False)
                and not arbitrated_low_margin
                and not margin_gate_hard
                and hasattr(lora_bank, "set_soft_routing")
            ):
                # Normalize probabilities if requested
                import torch
                raw_scores = list(decision.scores.values())
                adapters = list(decision.scores.keys())
                temp = float(getattr(router, "soft_routing_temperature", 1.0))
                if temp != 1.0 and len(raw_scores) > 0:
                    logits = torch.tensor(raw_scores, dtype=torch.float32) / max(temp, 1e-5)
                    probs = torch.softmax(logits, dim=0)
                    # Filter top-k
                    top_k = int(getattr(router, "soft_routing_top_k", 3))
                    if top_k > 0 and top_k < len(adapters):
                        top_vals, top_idx = torch.topk(probs, top_k)
                        probs = torch.zeros_like(probs)
                        probs[top_idx] = top_vals
                        probs = probs / probs.sum()
                    weights = probs.tolist()
                else:
                    weights = raw_scores

                if bool(getattr(router, "orthogonal_blend", False)):
                    weights = _orthogonal_gate_blend_weights(
                        adapters,
                        weights,
                        lora_bank,
                        float(getattr(router, "orthogonal_blend_lambda", 0.5)),
                    )
                    routing_stats["orthogonal_blend_count"] = routing_stats.get("orthogonal_blend_count", 0) + 1

                # Free unused branches when memory is tight
                import gc
                torch.cuda.empty_cache()
                gc.collect()

                lora_bank.set_soft_routing(adapters, weights)
            elif (
                getattr(router, "soft_routing", False)
                and not arbitrated_low_margin
                and not margin_gate_hard
                and hasattr(lora_bank, "blend_adapters")
            ):
                adapters = list(prob_scores.keys())
                weights = list(prob_scores.values())
                if bool(getattr(router, "orthogonal_blend", False)):
                    weights = _orthogonal_gate_blend_weights(
                        adapters,
                        weights,
                        lora_bank,
                        float(getattr(router, "orthogonal_blend_lambda", 0.5)),
                    )
                    routing_stats["orthogonal_blend_count"] = routing_stats.get("orthogonal_blend_count", 0) + 1
                lora_bank.blend_adapters(adapters, weights, "blended")
                lora_bank.set_active_adapter("blended")
            elif hasattr(lora_bank, "set_active_adapter"):
                lora_bank.set_active_adapter(decision.branch_name)
            
            preds.extend(model.generate([p], max_new_tokens=effective_max_new_tokens))
            
            # Clear soft routing
            if getattr(router, "soft_routing", False) and hasattr(lora_bank, "set_soft_routing"):
                lora_bank.set_soft_routing(None, None)
    else:
        if audited:
            gen_audit = model.generate_with_ids(prompts, max_new_tokens=effective_max_new_tokens)
            preds = [x.get("raw_generated_text", "") for x in gen_audit]
        else:
            gen_audit = None
            preds = model.generate(prompts, max_new_tokens=effective_max_new_tokens)
        routing_details = [{} for _ in preds]

    details: List[Dict[str, Any]] = []
    for example_idx, (ex, p, pred, y, routing_detail) in enumerate(zip(eval_examples, prompts, preds, targets, routing_details)):
        total += 1
        norm_pred = _normalize(pred, prompt=p, cfg=normalization_cfg)
        norm_gold = _normalize(y, prompt=p, cfg=normalization_cfg)
        matched = norm_pred == norm_gold
        token_f1 = _token_f1(norm_pred, norm_gold)
        lcs_overlap = _lcs_overlap(norm_pred, norm_gold)
        rouge_l = _rouge_l_fscore(norm_pred, norm_gold)
        bleu = _sentence_bleu4(norm_pred, norm_gold)
        slot_error = _dialogue_slot_error_rate(ex.input, norm_pred)
        task_score = _score_task_aware(
            pred=pred,
            gold=y,
            norm_pred=norm_pred,
            norm_gold=norm_gold,
            instruction=ex.instruction,
            input_text=ex.input,
            cfg=normalization_cfg,
        )
        bad_prefix = _starts_incorrectly(norm_pred, norm_gold)

        pred_tokens = [t for t in norm_pred.split() if t]
        gold_tokens = [t for t in norm_gold.split() if t]
        prefix_1_match = pred_tokens[:1] == gold_tokens[:1]
        prefix_3_match = pred_tokens[:3] == gold_tokens[:3]
        prefix_5_match = pred_tokens[:5] == gold_tokens[:5]

        teacher_forced_loss: Optional[float] = None
        teacher_forced_answer_token_acc: Optional[float] = None
        teacher_forced_num_loss_tokens: Optional[int] = None
        teacher_forced_num_supervised_label_tokens: Optional[int] = None
        if enable_teacher_forced_eval and total <= teacher_forced_eval_max_examples:
            # Teacher-forced forward pass over (prompt + gold assistant target).
            # Token accuracy must use the same shifted span as HF causal LM loss
            # (argmax(logits[:, :-1]) vs labels[:, 1:], ignoring -100); same-index
            # argmax vs labels was incorrect and inflated mismatch diagnostics.
            import torch

            was_training = getattr(model, "model", None).training if getattr(model, "model", None) is not None else False
            # Keep it in eval for determinism.
            if getattr(model, "model", None) is not None:
                model.model.eval()

            with torch.no_grad():
                backbone_cfg = getattr(model, "cfg", None)
                max_len = int(getattr(backbone_cfg, "max_seq_len", 2048))
                min_tgt = int(getattr(backbone_cfg, "min_target_tokens_for_loss", 1))
                labeling_mode = str(
                    normalization_cfg.get("teacher_forced_labeling_mode")
                    or (getattr(backbone_cfg, "train_labeling_mode", None) if backbone_cfg is not None else None)
                    or "manual"
                )
                completion_template = str(
                    normalization_cfg.get("completion_only_response_template")
                    or (
                        getattr(backbone_cfg, "completion_only_response_template", None)
                        if backbone_cfg is not None
                        else None
                    )
                    or "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                mask_eos = bool(normalization_cfg.get("mask_eos_token_in_labels", True))
                mask_spec = bool(normalization_cfg.get("mask_all_special_tokens_in_labels", True))

                enc = build_supervised_labels(
                    tok,
                    ex.instruction,
                    ex.input,
                    str(y),
                    max_len=max_len,
                    min_target_tokens=min_tgt,
                    mask_eos_token_in_labels=mask_eos,
                    mask_all_special_tokens_in_labels=mask_spec,
                    labeling_mode=labeling_mode,
                    completion_only_response_template=completion_template,
                )
                full_ids = enc.full_ids
                labels = enc.labels

                input_ids_t = torch.tensor([full_ids], dtype=torch.long, device=model.device)
                attention_mask_t = torch.ones_like(input_ids_t, dtype=torch.long)
                labels_t = torch.tensor([labels], dtype=torch.long, device=model.device)

                outputs = model.model(
                    input_ids=input_ids_t, attention_mask=attention_mask_t, labels=labels_t, return_dict=True
                )
                teacher_forced_loss = float(outputs.loss.detach().float().item())

                logits = outputs.logits  # [1, T, V]
                teacher_forced_answer_token_acc, _, n_loss = teacher_forced_token_accuracy_shifted(logits, labels_t)
                teacher_forced_num_loss_tokens = int(n_loss)
                teacher_forced_num_supervised_label_tokens = int(count_supervised_label_tokens(labels_t))

            if getattr(model, "model", None) is not None and was_training:
                model.model.train()

        if matched:
            correct += 1
        if bool(task_score["task_aware_match"]):
            task_aware_correct += 1
        details.append(
            {
                "instruction": ex.instruction,
                "input_text": ex.input,
                "gold_output": y,
                "source_segment_id": int(segment.segment_id),
                "source_segment_name": str(segment.segment_name),
                "source_example_idx": int(example_idx),
                "requested_max_new_tokens": int(max_new_tokens),
                "effective_max_new_tokens": int(effective_max_new_tokens),
                "raw_generated_output": pred,
                "normalized_prediction": norm_pred,
                "normalized_gold": norm_gold,
                "match": matched,
                "strict_match": matched,
                **task_score,
                "token_f1": float(token_f1),
                "lcs_overlap": float(lcs_overlap),
                "rouge_l": float(rouge_l),
                "bleu": float(bleu),
                "slot_error_rate": None if slot_error is None else float(slot_error),
                "bad_prefix_mismatch": bool(bad_prefix),
                "prefix_1_match": bool(prefix_1_match),
                "prefix_3_match": bool(prefix_3_match),
                "prefix_5_match": bool(prefix_5_match),
                "formatted_infer_prompt": p,
                "teacher_forced_loss": teacher_forced_loss,
                "teacher_forced_answer_token_acc": teacher_forced_answer_token_acc,
                "teacher_forced_num_loss_tokens": teacher_forced_num_loss_tokens,
                "teacher_forced_num_supervised_label_tokens": teacher_forced_num_supervised_label_tokens,
                **routing_detail,
            }
        )

        # Optional continuation boundary / slicing audit.
        if audited and len(details) <= infer_token_audit_max_examples:
            idx = len(details) - 1
            audit = gen_audit[idx]
            prompt_len = int(audit.get("infer_prompt_token_len", 0))
            gen_full_ids = audit.get("generated_full_ids", [])
            gen_cont_ids = audit.get("generated_continuation_ids", [])
            assert gen_cont_ids == gen_full_ids[prompt_len:], "Eval slice assertion failed"

            # Verify decoded prediction comes from continuation ids only.
            tok_redecoded = tok.decode(gen_cont_ids, skip_special_tokens=True) if gen_cont_ids else ""
            if tok_redecoded != (audit.get("raw_generated_text", "") or ""):
                # Don't hard-fail; just record for debugging.
                details[-1]["continuation_decoding_mismatch"] = True

            details[-1].update(
                {
                    "formatted_infer_prompt_text": p,
                    "infer_prompt_token_ids": audit.get("infer_prompt_token_ids", []),
                    "infer_prompt_token_len": prompt_len,
                    "generated_full_ids": gen_full_ids,
                    "generated_continuation_ids": gen_cont_ids,
                    "decoded_prompt_tail": audit.get("decoded_prompt_tail", ""),
                    "decoded_continuation_head": audit.get("decoded_continuation_head", ""),
                    "raw_generated_text": audit.get("raw_generated_text", ""),
                }
            )

    acc = correct / max(1, total)
    task_aware_acc = task_aware_correct / max(1, total)
    return float(acc), float(task_aware_acc), routing_stats, details


def _merge_routing_stats(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    dst["num_routed"] += int(src.get("num_routed", 0))
    for k, v in src.get("branch_counts", {}).items():
        dst["branch_counts"][k] = dst["branch_counts"].get(k, 0) + int(v)
    for k, v in src.get("oracle_best_branch_counts", {}).items():
        dst["oracle_best_branch_counts"][k] = dst["oracle_best_branch_counts"].get(k, 0) + int(v)
    dst["oracle_agreement_count"] += int(src.get("oracle_agreement_count", 0))
    dst["oracle_num_examples"] += int(src.get("oracle_num_examples", 0))
    dst["confidence_sum"] += float(src.get("confidence_sum", 0.0))
    dst["entropy_sum"] += float(src.get("entropy_sum", 0.0))
    dst["oracle_margin_sum"] += float(src.get("oracle_margin_sum", 0.0))


def _resolve_eval_max_new_tokens(
    *,
    max_new_tokens: int,
    eval_examples: List[Example],
    normalization_cfg: Dict[str, Any],
) -> int:
    """Use shorter greedy generations for structured short-answer evals, while keeping raw text."""
    if not bool(normalization_cfg.get("auto_short_answer_max_new_tokens", True)):
        return int(max_new_tokens)
    if not eval_examples:
        return int(max_new_tokens)
    short_limit = int(normalization_cfg.get("short_answer_max_new_tokens", 16))
    step_limit = int(normalization_cfg.get("step_answer_max_new_tokens", 8))
    golds = [str(ex.output or "") for ex in eval_examples]
    if all(_extract_after_step(g) is not None for g in golds):
        return int(min(max_new_tokens, step_limit))
    normalized_golds = [_basic_answer_normalize(g) for g in golds]
    max_gold_tokens = max((len(g.split()) for g in normalized_golds), default=0)
    if max_gold_tokens <= int(normalization_cfg.get("short_answer_gold_token_threshold", 6)):
        return int(min(max_new_tokens, short_limit))
    return int(max_new_tokens)


def _score_task_aware(
    *,
    pred: str,
    gold: str,
    norm_pred: str,
    norm_gold: str,
    instruction: str,
    input_text: str,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(cfg.get("enable_task_aware_score", True)):
        return {
            "task_aware_match": bool(norm_pred == norm_gold),
            "task_aware_score": float(norm_pred == norm_gold),
            "task_score_type": "strict_em",
            "extracted_prediction": "",
            "extracted_gold": "",
            "prediction_for_scoring": norm_pred,
        }

    pred_for_scoring = _truncate_prediction_for_scoring(pred, cfg)
    norm_pred_for_scoring = _basic_answer_normalize(pred_for_scoring)
    norm_gold_basic = _basic_answer_normalize(gold)

    pred_step = _extract_after_step(pred_for_scoring)
    gold_step = _extract_after_step(gold)
    if gold_step is not None:
        matched = pred_step == gold_step
        return {
            "task_aware_match": bool(matched),
            "task_aware_score": float(matched),
            "task_score_type": "after_step_extracted_em",
            "extracted_prediction": "" if pred_step is None else str(pred_step),
            "extracted_gold": str(gold_step),
            "prediction_for_scoring": norm_pred_for_scoring,
        }

    pred_label = _extract_label_like_answer(pred_for_scoring, gold, instruction=instruction, input_text=input_text)
    gold_label = _extract_label_like_answer(gold, gold, instruction=instruction, input_text=input_text)
    if gold_label:
        matched = pred_label == gold_label
        return {
            "task_aware_match": bool(matched),
            "task_aware_score": float(matched),
            "task_score_type": "label_accuracy",
            "extracted_prediction": pred_label or "",
            "extracted_gold": gold_label,
            "prediction_for_scoring": norm_pred_for_scoring,
        }

    matched = norm_pred == norm_gold
    return {
        "task_aware_match": bool(matched),
        "task_aware_score": float(matched),
        "task_score_type": "strict_em",
        "extracted_prediction": "",
        "extracted_gold": "",
        "prediction_for_scoring": norm_pred_for_scoring or norm_pred,
    }


def _truncate_prediction_for_scoring(text: str, cfg: Dict[str, Any]) -> str:
    out = str(text or "")
    if bool(cfg.get("score_truncate_at_first_blankline", True)):
        out = out.split("\n\n", 1)[0]
    if bool(cfg.get("score_truncate_at_first_newline", True)):
        out = out.split("\n", 1)[0]
        
    # Remove chatty prefixes often generated by instruction-tuned models
    prefixes = [
        "the correct answer is ",
        "the correct answer is: ",
        "the answer is ",
        "the answer is: ",
        "the output is ",
        "the output is: ",
        "the correct label is ",
        "sure, ",
        "sure! ",
        "here is ",
        "my answer is ",
        "the correct option is "
    ]
    out_lower = out.lower().strip()
    for prefix in prefixes:
        idx = out_lower.find(prefix)
        if idx != -1:
            # slice out from the end of the prefix
            out = out[idx + len(prefix):]
            out_lower = out.lower().strip()
            
    if bool(cfg.get("score_truncate_at_first_sentence_end", True)):
        m = re.search(r"(?<!\b[A-Z])[.!?](?:\s|$)", out)
        if m is not None:
            out = out[: m.end()]
    return out.strip()


def _extract_after_step(text: str) -> Optional[int]:
    m = re.search(r"\bafter\s+step\s*(\d+)\b", str(text or ""), flags=re.IGNORECASE)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_label_like_answer(text: str, gold: str, *, instruction: str, input_text: str) -> Optional[str]:
    gold_norm = _basic_answer_normalize(gold)
    if not gold_norm:
        return None

    # Numeric / yes-no / true-false / A-D answers are common short structured labels.
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", gold_norm):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
        return _basic_answer_normalize(m.group(0)) if m else ""
    if gold_norm in {"yes", "no", "true", "false"}:
        m = re.search(r"\b(yes|no|true|false)\b", str(text or ""), flags=re.IGNORECASE)
        return _basic_answer_normalize(m.group(1)) if m else ""
    if re.fullmatch(r"[a-d]", gold_norm):
        m = re.search(r"\b([A-Da-d])\b", str(text or ""))
        return _basic_answer_normalize(m.group(1)) if m else ""

    gold_tokens = gold_norm.split()
    if len(gold_tokens) > 4:
        return None

    first_chunk = _basic_answer_normalize(_truncate_prediction_for_scoring(str(text or ""), {}))
    if first_chunk == gold_norm:
        return gold_norm

    # Only credit contained labels for very short class names to avoid over-crediting free-form answers.
    context = f"{instruction}\n{input_text}".lower()
    looks_classification = any(k in context for k in ["label", "class", "category", "sentiment", "intent", "choose"])
    if looks_classification and re.search(rf"(?<!\w){re.escape(gold_norm)}(?!\w)", first_chunk):
        return gold_norm
    return first_chunk if looks_classification and len(first_chunk.split()) <= 4 else None


def _extract_instruction_from_prompt(prompt: str) -> str:
    # Legacy helper: preserved for compatibility with old debug files.
    marker = "指令："
    if marker in prompt:
        rest = prompt.split(marker, 1)[1]
        return rest.split("\n", 1)[0]
    return prompt


def _save_debug_examples(
    *,
    save_dir: str,
    segment_id: int,
    examples: List[Dict[str, Any]],
    normalization_cfg: Dict[str, Any],
    generation_cfg: Dict[str, Any],
) -> None:
    d = Path(save_dir)
    d.mkdir(parents=True, exist_ok=True)
    out = {
        "segment_id": segment_id,
        "normalization_cfg": normalization_cfg,
        "generation_cfg": generation_cfg,
        "examples": examples,
    }
    (d / f"eval_segment_{segment_id:03d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

