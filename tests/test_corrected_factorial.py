"""The published factorial and manifest stay inside the repository."""
from __future__ import annotations

import csv
import unittest
from pathlib import Path

import yaml

from core.replay_budget import validate_v3c_replay_block


ROOT = Path(__file__).resolve().parents[1]
FACTORIAL = ROOT / "configs" / "experiments" / "corrected_factorial.yaml"
MANIFEST = ROOT / "results" / "manifest.csv"


class CorrectedFactorialTests(unittest.TestCase):
    def test_factorial_describes_the_finished_matrix(self) -> None:
        text = FACTORIAL.read_text(encoding="utf-8")
        self.assertNotIn("/root/", text)
        self.assertIn("${DATA_ROOT}", text)
        self.assertIn("${RESULTS_ROOT}", text)
        payload = yaml.safe_load(text)
        self.assertEqual(payload["n_runs"], 36)
        self.assertEqual(payload["holdout_seed"], 456)
        self.assertEqual(payload["mixed_batch"]["rho"], 0.20)
        self.assertEqual(payload["mixed_batch"]["budget_policy"], "unified")
        self.assertEqual(payload["mixed_batch"]["monitor"], "train_holdout")
        self.assertEqual(payload["paths"]["data_root"], "${DATA_ROOT}")
        self.assertEqual(payload["paths"]["results_root"], "${RESULTS_ROOT}")
        runs = payload["runs"]
        self.assertEqual(len(runs), 36)
        self.assertEqual(len({row["id"] for row in runs}), 36)
        self.assertEqual({row["selector"] for row in runs}, {"random", "compat_uniform", "true_prototype"})
        self.assertEqual({row["objective"] for row in runs}, {"ce", "opsd"})
        self.assertEqual({row["stream"] for row in runs}, {"instrdialog", "instrdialogpp"})
        self.assertEqual({row["seed"] for row in runs}, {456, 789, 42})
        self.assertEqual({row["lambda_opsd"] for row in runs if row["objective"] == "ce"}, {0.0})
        self.assertEqual({row["lambda_opsd"] for row in runs if row["objective"] == "opsd"}, {0.2})
        for row in runs:
            self.assertEqual(row["holdout_seed"], 456)
            self.assertEqual(row["rho"], 0.20)
            self.assertEqual(row["budget_policy"], "unified")
            self.assertEqual(row["monitor"], "train_holdout")
            config = ROOT / row["config"]
            self.assertTrue(config.is_file(), row["config"])
            self.assertFalse(Path(row["config"]).is_absolute())

    def test_manifest_paths_are_repository_relative(self) -> None:
        text = MANIFEST.read_text(encoding="utf-8")
        self.assertNotIn("/root/", text)
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        v3c = [row for row in rows if row["run_id"].startswith("v3c_")]
        self.assertEqual(len(v3c), 36)
        for row in rows:
            for key in ("config_path", "checkpoint_path", "metrics_path", "prediction_path"):
                value = row[key]
                self.assertFalse(value.startswith("/"), f"{row['run_id']} {key}")
                if value:
                    self.assertTrue((ROOT / value).exists(), value)

    def test_queue_refusal_still_rejects_an_incomplete_block(self) -> None:
        with self.assertRaises(ValueError):
            validate_v3c_replay_block({"strategy": "random", "budget_policy": "unified"})
        with self.assertRaises(ValueError):
            validate_v3c_replay_block(
                {
                    "strategy": "true_prototype",
                    "monitor": "eval_split_monitor",
                    "holdout_seed": 456,
                    "holdout_n": 5,
                    "budget_policy": "unified",
                    "ratio_definition": "mixed_batch",
                    "target_fraction": 0.20,
                }
            )


if __name__ == "__main__":
    unittest.main()
