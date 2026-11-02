"""Mixed-batch replay budget. Training seed is not an input."""
from __future__ import annotations

import math
import random
import unittest

from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod, SegmentMemory
from core.data import Example, Segment
from core.replay_budget import (
    allocate_mixed_batch,
    half_up,
    largest_remainder,
    mixed_batch_n_replay,
    realized_replay_fraction,
    validate_v3c_replay_block,
)
from methods.pgdr import MONITOR_TRAIN_HOLDOUT, quota, replay_quota, split_train_holdout


def _ex(seg: int, i: int) -> Example:
    return Example(
        instruction=f"task-{seg}",
        input=f"row-{seg}-{i}",
        output=f"gold-{seg}-{i}",
        outputs=[f"gold-{seg}-{i}"],
    )


def _cfg(strategy: str, seed: int = 456, monitor: str = "eval_split_monitor") -> dict:
    return {
        "seed": seed,
        "replay": {
            "strategy": strategy,
            "replay_ratio": 0.2,
            "buffer_k": 10,
            "neighbor_ratio": 0.1,
            "forgotten_ratio": 0.1,
            "transfer_slot": "neighbor",
            "monitor": monitor,
            "holdout_n": 5,
            "holdout_seed": 456,
            "budget_policy": "unified",
            "ratio_definition": "mixed_batch",
            "target_fraction": 0.20,
        },
        "teacher_proto": {"mode": "none"},
        "opsd": {"divergence": "jsd", "beta": 0.5},
        "ce_opsd": {"lambda_opsd": 0.0},
        "eval_normalization": {"format_normalize": True},
        "model": {"gen_max_new_tokens": 8},
    }


def _memory(sid: int, n: int = 10, *, score: str = "strict_em") -> SegmentMemory:
    return SegmentMemory(
        segment_id=sid,
        instruction=f"task-{sid}",
        task_score_type=score,
        mean_gold_len=1.0,
        embedding=[1.0, 0.0],
        examples=[_ex(sid, i) for i in range(n)],
    )


def _current(sid: int = 9) -> SegmentMemory:
    return SegmentMemory(sid, "cur", "strict_em", 1.0, [1.0, 0.0], [])


class BudgetMathTests(unittest.TestCase):
    def test_half_up_is_not_bankers_rounding(self) -> None:
        self.assertEqual(half_up(4.5), 5)
        self.assertEqual(half_up(12.5), 13)
        self.assertEqual(half_up(11.25), 11)
        self.assertEqual(round(4.5), 4)
        self.assertEqual(round(12.5), 12)

    def test_45_and_50_at_four_rhos(self) -> None:
        expected = {
            (45, 0.0): 0,
            (45, 0.1): 5,
            (45, 0.2): 11,
            (45, 0.4): 30,
            (50, 0.0): 0,
            (50, 0.1): 6,
            (50, 0.2): 13,
            (50, 0.4): 33,
        }
        for (n_current, rho), n_replay in expected.items():
            self.assertEqual(mixed_batch_n_replay(n_current, rho), n_replay)
            budget = allocate_mixed_batch(n_current, rho)
            self.assertEqual(budget.n_replay_target, n_replay)
            self.assertEqual(budget.n_related + budget.n_degraded, n_replay)
            self.assertEqual(
                budget.realized_replay_fraction,
                realized_replay_fraction(n_replay, n_current),
            )

    def test_45_at_rho_020_is_11_of_56_not_exactly_20_percent(self) -> None:
        budget = allocate_mixed_batch(45, 0.20)
        self.assertEqual(budget.n_replay_target, 11)
        self.assertEqual(budget.n_related, 6)
        self.assertEqual(budget.n_degraded, 5)
        self.assertTrue(math.isclose(budget.realized_replay_fraction, 11 / 56))
        self.assertFalse(math.isclose(budget.realized_replay_fraction, 0.20))
        self.assertAlmostEqual(budget.realized_replay_fraction, 0.19642857142857142)

    def test_odd_quota_gives_the_tie_to_related(self) -> None:
        self.assertEqual(largest_remainder(11, ["related", "degraded"]), {"related": 6, "degraded": 5})
        self.assertEqual(largest_remainder(1, ["related", "degraded"]), {"related": 1, "degraded": 0})
        self.assertEqual(largest_remainder(5, ["related", "degraded"]), {"related": 3, "degraded": 2})
        self.assertEqual(largest_remainder(6, ["related", "degraded"]), {"related": 3, "degraded": 3})
        swapped = largest_remainder(11, ["degraded", "related"])
        self.assertEqual(swapped["degraded"], 6)
        self.assertEqual(swapped["related"], 5)

    def test_seed_is_not_an_argument_and_does_not_change_the_target(self) -> None:
        first = allocate_mixed_batch(45, 0.20)
        second = allocate_mixed_batch(45, 0.20)
        self.assertEqual(first, second)
        targets = []
        for seed in (1, 42, 456, 789, 99991):
            method = CeOpsdReplayMethod(_cfg("true_prototype", seed=seed))
            method._memories = [_memory(0, 20), _memory(1, 20)]
            method._forgetting_by_segment = {0: 0.2, 1: 0.8}
            method._current_memory = _current()
            random.seed(seed)
            method._mix_current_and_replay(Segment(3, "seg-3", [_ex(3, i) for i in range(45)], []))
            targets.append(method._last_replay_log["n_replay_target"])
            self.assertEqual(method._last_replay_log["n_replay_actual"], 11)
            self.assertEqual(method._last_replay_log["allocator_n_related"], 6)
            self.assertEqual(method._last_replay_log["allocator_n_degraded"], 5)
        self.assertEqual(set(targets), {11})

    def test_pool_shortfall_does_not_reround(self) -> None:
        method = CeOpsdReplayMethod(_cfg("random"))
        method._memories = [_memory(0, 3)]
        method._current_memory = _current()
        method._mix_current_and_replay(Segment(2, "seg-2", [_ex(2, i) for i in range(45)], []))
        log = method._last_replay_log
        self.assertEqual(log["n_replay_target"], 11)
        self.assertEqual(log["n_replay_actual"], 3)
        self.assertEqual(log["shortfall"], 8)
        self.assertEqual(log["n_unique"], 3)
        self.assertTrue(math.isclose(log["realized_replay_fraction"], 3 / 48))
        self.assertTrue(math.isclose(log["target_replay_fraction"], 11 / 56))

    def test_random_uniform_and_prototype_share_one_count(self) -> None:
        totals = {}
        for strategy in ("random", "compat_uniform", "true_prototype"):
            method = CeOpsdReplayMethod(_cfg(strategy))
            method._memories = [_memory(0, 12), _memory(1, 12), _memory(2, 12)]
            method._forgetting_by_segment = {0: 0.1, 1: 0.4, 2: 0.9}
            method._current_memory = _current()
            random.seed(456)
            method._mix_current_and_replay(Segment(4, "seg-4", [_ex(4, i) for i in range(45)], []))
            log = method._last_replay_log
            totals[strategy] = (
                log["n_replay_target"],
                log["n_replay_actual"],
                log["allocator_n_related"],
                log["allocator_n_degraded"],
            )
            self.assertEqual(log["n_related"] + log["n_degraded"] + (log["n_replay_actual"] if strategy == "random" else 0), log["n_replay_actual"] if strategy == "random" else log["n_related"] + log["n_degraded"])
            self.assertEqual(log["shortfall"], 0)
            per_sum = sum(int(row["n_replay"]) for row in log["per_segment"])
            self.assertEqual(per_sum, log["n_replay_actual"])
        self.assertEqual(len(set(totals.values())), 1)
        self.assertEqual(totals["random"][0], 11)
        random_method_related = totals["random"]
        self.assertEqual(random_method_related[2:], (6, 5))

    def test_related_and_degraded_do_not_reuse_the_same_row(self) -> None:
        method = CeOpsdReplayMethod(_cfg("true_prototype"))
        method._memories = [_memory(0, 4), _memory(1, 20)]
        method._forgetting_by_segment = {0: 0.95, 1: 0.1}
        method._current_memory = _current()
        random.seed(7)
        mixed = method._mix_current_and_replay(Segment(5, "seg-5", [_ex(5, i) for i in range(45)], []))
        log = method._last_replay_log
        self.assertEqual(log["n_replay_actual"], 11)
        self.assertEqual(log["n_unique"], 11)
        self.assertEqual(len(mixed), 56)
        self.assertEqual(len({(ex.instruction, ex.input) for ex in mixed}), 56)

    def test_legacy_proto_and_uniform_keep_the_old_round(self) -> None:
        self.assertEqual(quota(45, 0.1), 4)
        self.assertEqual(replay_quota(45, strategy="proto", replay_ratio=0.2, neighbor_ratio=0.1, forgotten_ratio=0.1)["n_replay"], 8)
        self.assertEqual(replay_quota(45, strategy="uniform", replay_ratio=0.2, neighbor_ratio=0.1, forgotten_ratio=0.1)["n_replay"], 9)
        for strategy, target in (("proto", 8), ("uniform", 9)):
            cfg = _cfg(strategy)
            cfg["replay"].pop("budget_policy")
            cfg["replay"].pop("ratio_definition")
            cfg["replay"].pop("target_fraction")
            method = CeOpsdReplayMethod(cfg)
            self.assertEqual(method.budget_policy, "legacy")
            method._memories = [_memory(0, 20)]
            method._forgetting_by_segment = {0: 0.5}
            method._current_memory = _current()
            method._mix_current_and_replay(Segment(6, "seg-6", [_ex(6, i) for i in range(45)], []))
            self.assertEqual(method._last_replay_log["n_replay_target"], target)
            self.assertEqual(method._last_replay_log["n_replay_actual"], target)
            self.assertNotEqual(method._last_replay_log["selector"], "l2_cosine_descending")

    def test_v3c_gate_refuses_missing_fields_and_accepts_the_pinned_block(self) -> None:
        good = _cfg("random", monitor=MONITOR_TRAIN_HOLDOUT)["replay"]
        validate_v3c_replay_block(good)
        broken = dict(good)
        del broken["budget_policy"]
        with self.assertRaises(ValueError):
            validate_v3c_replay_block(broken)
        legacy = {"strategy": "proto", "monitor": "train_holdout", "holdout_seed": 456, "holdout_n": 5}
        with self.assertRaises(ValueError):
            validate_v3c_replay_block(legacy)

    def test_holdout_and_future_rows_are_not_replayed(self) -> None:
        seg0 = Segment(0, "seg-0", [_ex(0, i) for i in range(50)], [_ex(90, i) for i in range(10)])
        seg1 = Segment(1, "seg-1", [_ex(1, i) for i in range(50)], [_ex(91, i) for i in range(10)])
        _, _, expected = split_train_holdout(seg0.train, segment_id=0, seed=456, holdout_n=5)
        for strategy in ("random", "compat_uniform", "true_prototype"):
            for train_seed in (456, 789, 42):
                method = CeOpsdReplayMethod(_cfg(strategy, seed=train_seed, monitor=MONITOR_TRAIN_HOLDOUT))
                self.assertEqual(method.holdout_seed, 456)
                method._training_examples(segment=seg0, model=_Stub(), lora=_StubLora())
                self.assertEqual(list(method._last_audit["holdout_indices"]), expected)
                self.assertEqual(method._last_replay_log["n_replay_target"], 0)
                method._memories.append(method._current_memory)
                mixed = method._training_examples(segment=seg1, model=_Stub(), lora=_StubLora())
                hold_ids = {id(ex) for rows in method._holdouts.values() for ex in rows}
                eval_ids = {id(ex) for ex in seg0.eval + seg1.eval}
                self.assertTrue(hold_ids.isdisjoint({id(ex) for ex in mixed}))
                self.assertTrue(eval_ids.isdisjoint({id(ex) for ex in mixed}))
                self.assertFalse(method._last_audit["holdout_in_train"])
                self.assertFalse(method._last_audit["eval_split_read_for_selection"])
                sources = [row["segment_id"] for row in method._last_replay_log["per_segment"]]
                self.assertTrue(all(int(sid) < 1 for sid in sources))
                self.assertEqual(method._last_replay_log["n_replay_target"], 11)
                self.assertNotIn(91, sources)
                before = dict(method._forgetting_by_segment)
                method.on_eval_end({"extra": {"forgetting_by_segment": [{"segment_id": 0, "forgetting": 1.0}]}})
                self.assertEqual(method._forgetting_by_segment, before)


class _Stub:
    def get_activations(self, prompts):
        return [[1.0, 0.0, 0.0] for _ in prompts]


class _StubLora:
    peft_model = None


if __name__ == "__main__":
    unittest.main()
