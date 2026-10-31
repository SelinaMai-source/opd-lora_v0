"""Replay quota, train-holdout leakage, JSD grads, and the 11-metric forgetting formula."""
from __future__ import annotations

import hashlib
import math
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from baselines.basic_baselines.ce_opsd.method import _gold_rollout_ids
from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod, SegmentMemory
from baselines.basic_baselines.opsd.method import _teacher_forward_logits, token_divergence
from core.data import Example, Segment, ContinualStream, apply_segment_order
from core.evaluate import _best_pred_gold_metrics
from core.metrics_utils import token_f1
from methods.opsd import ce_plus_opsd
from methods.opsd import token_divergence as opsd_token_divergence
from methods.pgdr import (
    MONITOR_EVAL_SPLIT,
    MONITOR_TRAIN_HOLDOUT,
    forgetting_from_accuracies,
    replay_quota,
    score_holdout_map,
    select_forgotten,
    split_train_holdout,
)

ROOT = Path("/root/research-repos/opd-lora_v3")
SOURCE_EVAL = Path("/root/autodl-tmp/rp_lora_v0/core/evaluate.py")

FEISHU_KEYS = (
    "seen_avg_score",
    "seen_avg_task_aware_score",
    "forgetting",
    "task_aware_forgetting",
    "token_f1_mean",
    "rouge_l_mean",
    "bleu_mean",
    "lcs_overlap_mean",
    "current_score",
    "current_task_aware_score",
    "task_aware_score_mean",
)


def _cfg(monitor: str, strategy: str, seed: int = 456, lambda_opsd: float = 0.2) -> dict:
    return {
        "seed": seed,
        "replay": {
            "strategy": strategy,
            "replay_ratio": 0.2,
            "buffer_k": 4,
            "neighbor_ratio": 0.1,
            "forgotten_ratio": 0.1,
            "transfer_slot": "neighbor",
            "monitor": monitor,
            "holdout_n": 5,
            "holdout_seed": seed,
        },
        "teacher_proto": {"mode": "none"},
        "opsd": {"divergence": "jsd", "beta": 0.5, "teacher_format_constraint": True},
        "ce_opsd": {"lambda_opsd": lambda_opsd},
        "eval_normalization": {"format_normalize": True},
        "model": {"gen_max_new_tokens": 8},
    }


def _ex(seg: int, i: int, split: str = "train") -> Example:
    return Example(
        instruction=f"task-{seg}-{split}",
        input=f"row-{i}",
        output=f"gold-{seg}-{i}",
        outputs=[f"gold-{seg}-{i}", f"alt-{seg}-{i}"],
    )


def _segment(sid: int, n_train: int = 50, n_eval: int = 10) -> Segment:
    return Segment(
        segment_id=sid,
        segment_name=f"seg-{sid}",
        train=[_ex(sid, i, "train") for i in range(n_train)],
        eval=[_ex(sid, i, "eval") for i in range(n_eval)],
    )


class _StubModel:
    def get_activations(self, prompts):
        return [[1.0, float(len(p) % 5), 0.25] for p in prompts]


class _StubLora:
    peft_model = None


class HoldoutTests(unittest.TestCase):
    def test_split_is_shared_across_the_four_methods(self) -> None:
        seg = _segment(3)
        indices = []
        for strategy, lam in (("uniform", 0.0), ("proto", 0.0), ("uniform", 0.2), ("proto", 0.2)):
            method = CeOpsdReplayMethod(_cfg(MONITOR_TRAIN_HOLDOUT, strategy, lambda_opsd=lam))
            mixed = method._training_examples(segment=seg, model=_StubModel(), lora=_StubLora())
            indices.append(tuple(method._last_audit["holdout_indices"]))
            self.assertEqual(len(method._current_holdout), 5)
            self.assertEqual(len(method._current_fit), 45)
            self.assertEqual(len(mixed), 45)
            self.assertFalse(method._last_audit["holdout_in_train"])
            self.assertFalse(method._last_audit["holdout_in_memory"])
            self.assertFalse(method._last_audit["eval_split_read_for_selection"])
            hold_ids = {id(ex) for ex in method._current_holdout}
            self.assertTrue(hold_ids.isdisjoint({id(ex) for ex in mixed}))
            self.assertTrue(hold_ids.isdisjoint({id(ex) for ex in method._current_memory.examples}))
        self.assertEqual(len(set(indices)), 1)
        self.assertEqual(len(indices[0]), 5)

    def test_split_depends_on_segment_and_seed_only(self) -> None:
        rows = [_ex(0, i) for i in range(50)]
        _, _, a = split_train_holdout(rows, segment_id=1, seed=456, holdout_n=5)
        _, _, b = split_train_holdout(rows, segment_id=1, seed=456, holdout_n=5)
        _, _, c = split_train_holdout(rows, segment_id=2, seed=456, holdout_n=5)
        _, _, d = split_train_holdout(rows, segment_id=1, seed=789, holdout_n=5)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_holdout_never_enters_later_replay_and_eval_is_unread(self) -> None:
        method = CeOpsdReplayMethod(_cfg(MONITOR_TRAIN_HOLDOUT, "proto"))
        seg0 = _segment(0)
        seg1 = _segment(1)
        method._training_examples(segment=seg0, model=_StubModel(), lora=_StubLora())
        method._memories.append(method._current_memory)
        method._forgetting_by_segment = {0: 0.8}
        mixed1 = method._training_examples(segment=seg1, model=_StubModel(), lora=_StubLora())
        hold_ids = {id(ex) for rows in method._holdouts.values() for ex in rows}
        eval_ids = {id(ex) for ex in seg0.eval + seg1.eval}
        self.assertTrue(hold_ids.isdisjoint({id(ex) for ex in mixed1}))
        self.assertTrue(eval_ids.isdisjoint({id(ex) for ex in mixed1}))
        self.assertFalse(method._last_audit["holdout_in_train"])
        self.assertFalse(method._last_audit["holdout_in_memory"])
        self.assertGreater(method._n_replay_used, 0)
        before = dict(method._forgetting_by_segment)
        method.on_eval_end(
            {
                "extra": {
                    "forgetting_by_segment": [
                        {"segment_id": 0, "forgetting": 0.99},
                        {"segment_id": 1, "forgetting": 0.5},
                    ]
                }
            }
        )
        self.assertEqual(method._forgetting_by_segment, before)
        self.assertFalse(method._eval_split_consumed)
        self.assertTrue(method._live_metrics is None or method._live_metrics.get("on_eval_end_ignored", True))

    def test_holdout_scorer_receives_holdout_rows_not_official_eval(self) -> None:
        captured = []

        def fake_eval_segment(*, segment, **_kwargs):
            captured.append(("train", [ex.input for ex in segment.train], [ex.input for ex in segment.eval]))
            self.assertEqual(segment.train, [])
            self.assertTrue(str(segment.segment_name).startswith("train_holdout_"))
            return 0.4, 0.4, {}, []

        import core.evaluate as evaluate

        original = evaluate._eval_segment
        evaluate._eval_segment = fake_eval_segment
        try:
            hold = {0: [_ex(0, 1), _ex(0, 2)], 1: [_ex(1, 3)]}
            accs = score_holdout_map(object(), hold, normalization_cfg={"format_normalize": True}, max_new_tokens=8)
        finally:
            evaluate._eval_segment = original
        self.assertEqual(accs, [(0, 0.4), (1, 0.4)])
        self.assertEqual(captured[0][2], ["row-1", "row-2"])
        self.assertNotIn("eval", " ".join(captured[0][2]))

    def test_eval_split_monitor_still_reads_on_eval_end(self) -> None:
        method = CeOpsdReplayMethod(_cfg(MONITOR_EVAL_SPLIT, "proto"))
        method._live_metrics = {}
        method.on_eval_end(
            {"extra": {"forgetting_by_segment": [{"segment_id": 4, "forgetting": 0.25}]}}
        )
        self.assertEqual(method._forgetting_by_segment[4], 0.25)
        self.assertTrue(method._eval_split_consumed)
        self.assertTrue(method._live_metrics["eval_split_read_for_selection"])
        self.assertFalse(method._live_metrics["on_eval_end_ignored"])

    def test_replay_quota_50(self) -> None:
        self.assertEqual(replay_quota(50, strategy="uniform", replay_ratio=0.2, neighbor_ratio=0.1, forgotten_ratio=0.1)["n_replay"], 10)
        proto = replay_quota(50, strategy="proto", replay_ratio=0.2, neighbor_ratio=0.1, forgotten_ratio=0.1)
        self.assertEqual(proto["n_neighbor"], 5)
        self.assertEqual(proto["n_forgotten"], 5)
        fit = replay_quota(45, strategy="proto", replay_ratio=0.2, neighbor_ratio=0.1, forgotten_ratio=0.1)
        self.assertEqual(fit["n_neighbor"], int(round(45 * 0.1)))
        self.assertEqual(fit["n_forgotten"], int(round(45 * 0.1)))

    def test_forgotten_selector_uses_passed_scores_and_version(self) -> None:
        mem_lo = SegmentMemory(0, "a", "strict_em", 1.0, [1.0, 0.0], [_ex(0, i) for i in range(4)])
        mem_hi = SegmentMemory(1, "b", "strict_em", 1.0, [0.0, 1.0], [_ex(1, i) for i in range(4)])
        picked = select_forgotten([mem_lo, mem_hi], {0: 0.1, 1: 0.9}, 3, selector_version=MONITOR_TRAIN_HOLDOUT)
        self.assertTrue(all(ex.instruction.startswith("task-1-") for ex in picked))
        self.assertEqual(len(picked), 3)
        fallback = select_forgotten([mem_lo, mem_hi], {0: 0.0, 1: 0.0}, 2, selector_version=MONITOR_EVAL_SPLIT)
        self.assertEqual(len(fallback), 2)
        with self.assertRaises(ValueError):
            select_forgotten([mem_lo], {0: 1.0}, 1, selector_version="eval_split")

    def test_method_proto_quota_on_50(self) -> None:
        method = CeOpsdReplayMethod(_cfg(MONITOR_EVAL_SPLIT, "proto"))
        method.buffer_k = 10
        mem = SegmentMemory(0, "a", "strict_em", 1.0, [1.0, 0.0, 0.0], [_ex(0, i) for i in range(10)])
        method._memories = [mem]
        method._forgetting_by_segment = {0: 0.4}
        method._current_memory = SegmentMemory(9, "cur", "strict_em", 1.0, [1.0, 0.0, 0.0], [])
        seg = _segment(9)
        mixed = method._mix_current_and_replay(seg)
        self.assertEqual(method._n_neighbor_used, 5)
        self.assertEqual(method._n_forgotten_used, 5)
        self.assertEqual(len(mixed), 60)


class OpsdTests(unittest.TestCase):
    def test_jsd_value_and_student_grad_teacher_frozen(self) -> None:
        teacher = torch.tensor([[10.0, -10.0]])
        student = torch.tensor([[-10.0, 10.0]], requires_grad=True)
        loss = opsd_token_divergence(teacher, student, kind="jsd", beta=0.5)
        self.assertTrue(math.isclose(float(loss.detach()), math.log(2.0), rel_tol=1e-4, abs_tol=1e-4))
        same = opsd_token_divergence(teacher, teacher.detach().clone().requires_grad_(True), kind="jsd", beta=0.5)
        self.assertLess(float(same.detach()), 1e-5)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

        student_net = nn.Linear(3, 4)
        teacher_net = nn.Linear(3, 4)
        for p in teacher_net.parameters():
            p.requires_grad_(False)
        x = torch.randn(2, 3)
        s_logits = student_net(x)
        with torch.no_grad():
            t_logits = teacher_net(x)
        jsd = opsd_token_divergence(t_logits, s_logits, kind="jsd", beta=0.5)
        jsd.backward()
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in student_net.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher_net.parameters()))

        wrapped = token_divergence(t_logits.detach(), s_logits.detach(), kind="jsd", beta=0.5)
        direct = opsd_token_divergence(t_logits.detach(), s_logits.detach(), kind="jsd", beta=0.5)
        self.assertTrue(torch.allclose(wrapped, direct))

    def test_teacher_forward_disables_adapter_and_has_no_grad(self) -> None:
        class _Adapter:
            def __init__(self) -> None:
                self.depth = 0
                self.entered = 0

            def disable_adapter(self):
                return self

            def __enter__(self):
                self.depth += 1
                self.entered += 1
                return self

            def __exit__(self, exc_type, exc, tb):
                self.depth -= 1
                return False

        adapter = _Adapter()

        class _Lora:
            peft_model = adapter

            def get_active_adapter_name(self):
                return "default"

        class _Model:
            def __init__(self) -> None:
                self.weight = torch.nn.Parameter(torch.ones(3))
                self.seen_depth = 0

            def forward_logits(self, ids, mask):
                self.seen_depth = adapter.depth
                return self.weight.view(1, 1, 3) * torch.ones(ids.shape[0], ids.shape[1], 3)

        model = _Model()
        ids = torch.ones(1, 2, dtype=torch.long)
        mask = torch.ones(1, 2, dtype=torch.long)
        out = _teacher_forward_logits(
            model,
            _Lora(),
            ids,
            mask,
            teacher_source="base",
            teacher_adapter="teacher",
            peft_model=adapter,
        )
        self.assertEqual(adapter.entered, 1)
        self.assertEqual(model.seen_depth, 1)
        self.assertFalse(out.requires_grad)
        ce = torch.nn.functional.mse_loss(torch.zeros(1, requires_grad=True), torch.ones(1))
        combined = ce_plus_opsd(ce, torch.tensor(0.5), 0.2)
        self.assertTrue(math.isclose(float(combined.detach()), float(ce.detach()) + 0.1, rel_tol=1e-5))


class MetricFormulaTests(unittest.TestCase):
    def test_evaluate_file_matches_source_tree(self) -> None:
        src = hashlib.sha256(SOURCE_EVAL.read_bytes()).hexdigest()
        dst = hashlib.sha256((ROOT / "core" / "evaluate.py").read_bytes()).hexdigest()
        self.assertEqual(src, dst)

    def test_multigold_and_forgetting_formula(self) -> None:
        pair = _best_pred_gold_metrics(
            pred="gold-0-1",
            golds=["nope", "gold-0-1"],
            prompt="",
            instruction="task",
            input_text="row",
            normalization_cfg={"format_normalize": True},
        )
        self.assertTrue(pair["match"])
        self.assertEqual(pair["selected_gold_index"], 1)
        ref = token_f1("cat dog", "dog cat")
        self.assertTrue(math.isclose(ref, 1.0))
        pair_f1 = _best_pred_gold_metrics(
            pred="cat dog",
            golds=["dog cat"],
            prompt="",
            instruction="task",
            input_text="row",
            normalization_cfg={"format_normalize": True},
        )
        self.assertFalse(pair_f1["match"])
        self.assertTrue(math.isclose(float(pair_f1["token_f1"]), ref))

        rows0, mean0, best0 = forgetting_from_accuracies([(0, 1.0)])
        self.assertEqual(mean0, 0.0)
        self.assertEqual(rows0[0]["forgetting"], 0.0)
        rows1, mean1, best1 = forgetting_from_accuracies([(0, 0.4), (1, 1.0)], best0)
        self.assertTrue(math.isclose(mean1, 0.6))
        self.assertTrue(math.isclose(rows1[0]["forgetting"], 0.6))
        self.assertEqual(rows1[1]["forgetting"], 0.0)
        _rows2, mean2, _best2 = forgetting_from_accuracies([(0, 0.8), (1, 0.5), (2, 1.0)], best1)
        self.assertTrue(math.isclose(mean2, 0.35))

    def test_feishu_eleven_keys(self) -> None:
        self.assertEqual(len(FEISHU_KEYS), 11)
        sample = {f"eval.{key}": 0.0 for key in FEISHU_KEYS}
        missing = [key for key in FEISHU_KEYS if f"eval.{key}" not in sample]
        self.assertEqual(missing, [])


class PendingExtensionTests(unittest.TestCase):
    def test_mix_15_5_and_5_15_quota(self) -> None:
        a = replay_quota(45, strategy="proto", replay_ratio=0.2, neighbor_ratio=0.15, forgotten_ratio=0.05)
        self.assertEqual(a["n_neighbor"], 7)
        self.assertEqual(a["n_forgotten"], 2)
        self.assertEqual(a["n_replay"], 9)
        b = replay_quota(45, strategy="proto", replay_ratio=0.2, neighbor_ratio=0.05, forgotten_ratio=0.15)
        self.assertEqual(b["n_neighbor"], 2)
        self.assertEqual(b["n_forgotten"], 7)

    def test_replay_ratio_zero_does_not_sample_memory(self) -> None:
        method = CeOpsdReplayMethod(_cfg(MONITOR_EVAL_SPLIT, "proto"))
        method.replay_ratio = 0.0
        method.neighbor_ratio = 0.0
        method.forgotten_ratio = 0.0
        method._memories = [
            SegmentMemory(0, "a", "strict_em", 1.0, [1.0, 0.0], [_ex(0, i) for i in range(10)])
        ]
        method._forgetting_by_segment = {0: 0.9}
        mixed = method._mix_current_and_replay(_segment(1, n_train=8))
        self.assertEqual(len(mixed), 8)
        self.assertEqual(method._n_replay_used, 0)
        self.assertEqual(method._n_forgotten_used, 0)

    def test_segment_order_rewrites_ids_without_mixing_examples(self) -> None:
        segs = [_segment(i, n_train=3, n_eval=2) for i in range(4)]
        stream = ContinualStream("bench", "v", segs)
        out = apply_segment_order(stream, [3, 0, 2, 1])
        self.assertEqual([s.segment_id for s in out.stream], [0, 1, 2, 3])
        self.assertEqual([s.segment_name for s in out.stream], ["seg-3", "seg-0", "seg-2", "seg-1"])
        self.assertEqual(out.stream[0].train[0].instruction, "task-3-train")
        self.assertEqual(out.stream[1].eval[0].instruction, "task-0-eval")
        with self.assertRaises(ValueError):
            apply_segment_order(stream, [0, 1, 1, 2])

    def test_gold_rollout_ids_are_tokenizer_continuation(self) -> None:
        class _Tok:
            def __call__(self, text, add_special_tokens=False):
                self.seen = (text, add_special_tokens)
                return {"input_ids": [11, 12, 13]}

        tok = _Tok()
        self.assertEqual(_gold_rollout_ids(tok, "hello"), [11, 12, 13])
        self.assertEqual(tok.seen, ("hello", False))


if __name__ == "__main__":
    unittest.main()
