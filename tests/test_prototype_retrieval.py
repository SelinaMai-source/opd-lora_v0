"""True prototype ranks explicit L2 cosine. compat_uniform does not."""
from __future__ import annotations

import math
import random
import unittest

from baselines.basic_baselines.ce_opsd_replay.method import CeOpsdReplayMethod, SegmentMemory, _cosine
from core.data import Example, Segment
from core.replay_budget import cosine_similarity


def _ex(seg: int, i: int) -> Example:
    return Example(
        instruction=f"task-{seg}",
        input=f"row-{seg}-{i}",
        output=f"gold-{seg}-{i}",
        outputs=[f"gold-{seg}-{i}", f"alt-{seg}-{i}"],
    )


def _unit(cosine: float) -> list:
    return [float(cosine), math.sqrt(max(0.0, 1.0 - float(cosine) * float(cosine)))]


def _cfg(strategy: str, seed: int = 456) -> dict:
    return {
        "seed": seed,
        "replay": {
            "strategy": strategy,
            "replay_ratio": 0.2,
            "buffer_k": 10,
            "neighbor_ratio": 0.1,
            "forgotten_ratio": 0.1,
            "monitor": "train_holdout",
            "holdout_n": 5,
            "holdout_seed": 456,
            "budget_policy": "unified",
            "ratio_definition": "mixed_batch",
            "target_fraction": 0.20,
        },
        "teacher_proto": {"mode": "none"},
        "ce_opsd": {"lambda_opsd": 0.2},
        "eval_normalization": {"format_normalize": True},
    }


def _arm_memories() -> tuple:
    """A/B/C cosines are 0.90/0.40/-0.20 only after L2 normalization.

    A is scaled down and B is scaled up, so a raw dot product ranks B first.
    """
    current = [1.0, 0.0]
    vectors = {
        "A": [value * 0.01 for value in _unit(0.90)],
        "B": [value * 50.0 for value in _unit(0.40)],
        "C": [value * 3.0 for value in _unit(-0.20)],
    }
    ids = {"A": 10, "B": 1, "C": 2}
    memories = []
    for name in ("B", "C", "A"):
        sid = ids[name]
        memories.append(
            SegmentMemory(
                segment_id=sid,
                instruction=f"task-{name}",
                task_score_type="strict_em",
                mean_gold_len=2.0,
                embedding=vectors[name],
                examples=[_ex(sid, i) for i in range(4)],
            )
        )
    current_memory = SegmentMemory(8, "current", "strict_em", 2.0, current, [])
    return current_memory, memories, ids


class PrototypeRetrievalTests(unittest.TestCase):
    def test_explicit_l2_cosine_beats_raw_dot(self) -> None:
        current, memories, ids = _arm_memories()
        by_id = {mem.segment_id: mem for mem in memories}
        self.assertAlmostEqual(cosine_similarity(current.embedding, by_id[ids["A"]].embedding), 0.90)
        self.assertAlmostEqual(cosine_similarity(current.embedding, by_id[ids["B"]].embedding), 0.40)
        self.assertAlmostEqual(cosine_similarity(current.embedding, by_id[ids["C"]].embedding), -0.20)
        raw = {
            name: _cosine(current.embedding, by_id[ids[name]].embedding)
            for name in ("A", "B", "C")
        }
        self.assertGreater(raw["B"], raw["A"])

    def test_true_prototype_takes_A_first_and_compat_uniform_does_not(self) -> None:
        current, memories, ids = _arm_memories()
        chosen_u = []
        for seed in range(40):
            random.seed(seed)
            uniform = CeOpsdReplayMethod(_cfg("compat_uniform", seed=seed))
            uniform._current_memory = current
            uniform._memories = memories
            uniform._forgetting_by_segment = {ids["A"]: 0.1, ids["B"]: 0.2, ids["C"]: 0.3}
            uniform._mix_current_and_replay(Segment(8, "cur", [_ex(8, i) for i in range(4)], []))
            log_u = uniform._last_replay_log
            self.assertEqual(log_u["selector"], "format_compatible_uniform")
            self.assertIn("not a cosine prototype", log_u["selector_note"])
            self.assertEqual(log_u["ranked_segment_ids"], [1, 2, 10])
            self.assertEqual(log_u["n_replay_target"], 1)
            self.assertEqual(log_u["n_related"], 1)
            self.assertEqual(log_u["allocator_n_degraded"], 0)
            self.assertEqual(log_u["related_source_segment_ids"], [log_u["related_source_segment_ids"][0]])
            chosen_u.append(log_u["related_source_segment_ids"][0])
            self.assertFalse(log_u["selector_l2_renormalizes"])

            random.seed(seed)
            proto = CeOpsdReplayMethod(_cfg("true_prototype", seed=seed))
            proto._current_memory = current
            proto._memories = memories
            proto._forgetting_by_segment = {ids["A"]: 0.1, ids["B"]: 0.2, ids["C"]: 0.3}
            proto._mix_current_and_replay(Segment(8, "cur", [_ex(8, i) for i in range(4)], []))
            log_p = proto._last_replay_log
            self.assertEqual(log_p["selector"], "l2_cosine_descending")
            self.assertEqual(log_p["ranked_segment_ids"], [ids["A"], ids["B"], ids["C"]])
            self.assertEqual(log_p["related_source_segment_ids"], [ids["A"]])
            self.assertEqual(log_p["n_replay_target"], log_u["n_replay_target"])
            self.assertEqual(log_p["n_replay_actual"], 1)
            self.assertTrue(log_p["selector_l2_renormalizes"])
            self.assertAlmostEqual(log_p["selected_similarity_max"], 0.90)
            self.assertAlmostEqual(log_p["selected_similarity_min"], 0.90)
            self.assertEqual(log_p["fallback_count"], 0)
            self.assertNotEqual(log_p["ranked_segment_ids"], log_u["ranked_segment_ids"])
        self.assertGreater(len(set(chosen_u)), 1)
        self.assertIn(ids["A"], chosen_u)
        self.assertTrue(any(sid != ids["A"] for sid in chosen_u))

    def test_ties_break_by_ascending_segment_id_then_lower_segments_fill_first(self) -> None:
        current = SegmentMemory(0, "cur", "strict_em", 1.0, [1.0, 0.0], [])
        high = SegmentMemory(5, "high", "strict_em", 1.0, [1.0, 0.0], [_ex(5, i) for i in range(3)])
        low = SegmentMemory(2, "low", "strict_em", 1.0, [3.0, 0.0], [_ex(2, i) for i in range(3)])
        method = CeOpsdReplayMethod(_cfg("true_prototype"))
        method._current_memory = current
        method._memories = [high, low]
        method._forgetting_by_segment = {}
        random.seed(0)
        method._mix_current_and_replay(Segment(7, "cur", [_ex(7, i) for i in range(4)], []))
        log = method._last_replay_log
        self.assertEqual(log["ranked_segment_ids"], [2, 5])
        self.assertEqual(log["related_source_segment_ids"], [2])

    def test_fills_A_before_B_and_records_quota_and_dedup(self) -> None:
        current, memories, ids = _arm_memories()
        # 20 current rows -> 5 replay = 3 related + 2 degraded.
        method = CeOpsdReplayMethod(_cfg("true_prototype"))
        method._current_memory = current
        short_a = next(mem for mem in memories if mem.segment_id == ids["A"])
        short_a.examples = short_a.examples[:2]
        method._memories = memories
        method._forgetting_by_segment = {ids["A"]: 0.99, ids["B"]: 0.01, ids["C"]: 0.01}
        random.seed(123)
        method._mix_current_and_replay(Segment(8, "cur", [_ex(8, i) for i in range(20)], []))
        log = method._last_replay_log
        self.assertEqual(log["n_replay_target"], 5)
        self.assertEqual(log["allocator_n_related"], 3)
        self.assertEqual(log["allocator_n_degraded"], 2)
        self.assertEqual(log["n_related"], 3)
        self.assertEqual(log["n_degraded"], 2)
        self.assertEqual(log["n_unique"], 5)
        self.assertEqual(log["fallback_count"], 0)
        self.assertEqual(log["related_source_segment_ids"].count(ids["A"]), 2)
        self.assertEqual(log["related_source_segment_ids"].count(ids["B"]), 1)
        self.assertNotIn(ids["C"], log["related_source_segment_ids"])
        scores = {row["segment_id"]: row["cosine"] for row in log["candidate_cosine_scores"]}
        self.assertAlmostEqual(scores[ids["A"]], 0.90)
        self.assertAlmostEqual(scores[ids["B"]], 0.40)
        self.assertAlmostEqual(scores[ids["C"]], -0.20)

    def test_fallback_when_compatible_memory_is_short(self) -> None:
        method = CeOpsdReplayMethod(_cfg("true_prototype"))
        method._current_memory = SegmentMemory(3, "cur", "strict_em", 1.0, [1.0, 0.0], [])
        method._memories = [
            SegmentMemory(0, "other", "generation", 1.0, [1.0, 0.0], [_ex(0, i) for i in range(6)]),
        ]
        method._forgetting_by_segment = {0: 0.4}
        random.seed(1)
        method._mix_current_and_replay(Segment(3, "cur", [_ex(3, i) for i in range(20)], []))
        log = method._last_replay_log
        self.assertEqual(log["ranked_segment_ids"], [])
        self.assertEqual(log["fallback_count"], 3)
        self.assertEqual(log["n_related"], 3)
        self.assertTrue(all(row["format_compatible"] is False for row in log["candidate_cosine_scores"]))

    def test_retrieval_uses_cached_vectors_and_ignores_gold_eval_and_future(self) -> None:
        model = _CountingModel()
        lora = _CountingLora()
        method = CeOpsdReplayMethod(_cfg("true_prototype"))
        train = [_ex(0, i) for i in range(8)]
        future = _ex(5, 0)
        official = _ex(80, 0)
        seg0 = Segment(0, "seg-0", train, [official])
        method._current_memory = method._build_memory(seg0, model, lora, source_examples=train[:6])
        self.assertEqual(model.calls, 1)
        self.assertEqual(lora.entered, 1)
        method._memories = [method._current_memory]
        calls_after_build = model.calls
        method._current_memory = SegmentMemory(1, "now", "strict_em", 1.0, [1.0, 0.0, 0.0], [])
        for ex in method._memories[0].examples:
            ex.output = "MUTATED-GOLD"
            ex.outputs = ["MUTATED-GOLD"]
        random.seed(4)
        mixed = method._mix_current_and_replay(
            Segment(1, "seg-1", [_ex(1, i) for i in range(20)], [official, future])
        )
        self.assertEqual(model.calls, calls_after_build)
        self.assertEqual(lora.entered, 1)
        blob = " ".join(ex.instruction + " " + ex.output for ex in mixed)
        self.assertNotIn("task-5", blob)
        self.assertNotIn("task-80", blob)
        self.assertTrue(all(int(row["segment_id"]) < 1 for row in method._last_replay_log["per_segment"]))
        self.assertNotIn("MUTATED-GOLD", " ".join(str(row) for row in method._last_replay_log["ranked_segment_ids"]))

    def test_random_ignores_cosine_and_forgetting(self) -> None:
        current, memories, ids = _arm_memories()
        seen = set()
        for seed in range(30):
            random.seed(seed)
            method = CeOpsdReplayMethod(_cfg("random", seed=seed))
            method._current_memory = current
            method._memories = memories
            method._forgetting_by_segment = {ids["C"]: 5.0}
            method._mix_current_and_replay(Segment(8, "cur", [_ex(8, i) for i in range(20)], []))
            log = method._last_replay_log
            self.assertEqual(log["selector"], "uniform_all_history")
            self.assertEqual(log["n_related"], 0)
            self.assertEqual(log["n_degraded"], 0)
            self.assertEqual(log["n_replay_target"], 5)
            self.assertEqual(log["allocator_n_related"], 3)
            seen.update(row["segment_id"] for row in log["per_segment"])
        self.assertEqual(seen, {ids["A"], ids["B"], ids["C"]})


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def get_activations(self, prompts):
        self.calls += 1
        return [[1.0, float(len(p) % 3), 0.0] for p in prompts]


class _Adapter:
    def __init__(self) -> None:
        self.entered = 0

    def disable_adapter(self):
        return self

    def __enter__(self):
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _CountingLora:
    def __init__(self) -> None:
        self.peft_model = _Adapter()

    @property
    def entered(self) -> int:
        return self.peft_model.entered


if __name__ == "__main__":
    unittest.main()
