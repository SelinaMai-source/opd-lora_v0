"""Benchmark checks on synthetic examples. No json file and no private disk."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks._common import resolve_stream_file, validate_dataset
from benchmarks.instrdialog import get_metadata as id_meta
from benchmarks.instrdialog import load_benchmark
from core.data import ContinualStream, Example, Segment


def _ex(instruction: str, row: str, *golds: str) -> Example:
    return Example(instruction=instruction, input=row, output=golds[0], outputs=list(golds))


def _stream() -> ContinualStream:
    return ContinualStream(
        benchmark="synthetic",
        version="test",
        stream=[
            Segment(
                segment_id=0,
                segment_name="alpha",
                train=[_ex("task-a", "train-0", "gold-a")],
                eval=[_ex("task-a", "eval-0", "gold-a", "alt-a")],
            ),
            Segment(
                segment_id=1,
                segment_name="beta",
                train=[_ex("task-b", "train-1", "gold-b")],
                eval=[_ex("task-b", "eval-1", "gold-b")],
            ),
        ],
    )


class BenchmarkIsolationTests(unittest.TestCase):
    def test_synthetic_stream_validates_without_a_dataset_file(self) -> None:
        stats = validate_dataset(_stream())
        self.assertEqual(stats, {"n_segments": 2, "n_train": 2, "n_eval": 2})

    def test_train_eval_overlap_is_rejected(self) -> None:
        shared = _ex("task-a", "same", "gold")
        stream = ContinualStream(
            benchmark="synthetic",
            version="test",
            stream=[Segment(0, "alpha", [shared], [shared])],
        )
        with self.assertRaises(ValueError):
            validate_dataset(stream)

    def test_metadata_and_missing_file_stay_on_the_given_root(self) -> None:
        meta = id_meta()
        self.assertTrue(meta["filename"].endswith("_multiref.json"))
        self.assertNotIn("autodl-tmp", str(meta))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                load_benchmark(root=tmp)
            self.assertIn(str(Path(tmp)), str(caught.exception))
            resolved = resolve_stream_file("instrdialog", meta["filename"], root=tmp)
            self.assertEqual(resolved, Path(tmp) / meta["filename"])


if __name__ == "__main__":
    unittest.main()
