import json
import tempfile
import unittest
from pathlib import Path

from core.run_guard import fingerprint, fingerprints_match, refuse_or_skip_existing_run


def _cfg(seed: int = 456) -> dict:
    return {
        "seed": seed,
        "baseline_name": "ce_opsd_replay",
        "paths": {
            "processed_stream_dir": "benchmark/instrdialog",
            "processed_stream_file": "citb_cl_dialogue_tasks_train50_eval10_multiref.json",
            "results_dir": "results",
        },
        "data": {"max_segments": -1},
        "ce_opsd": {"lambda_opsd": 0.2},
        "eval_normalization": {"format_normalize": True},
        "replay": {
            "strategy": "proto",
            "monitor": "train_holdout",
            "replay_ratio": 0.2,
            "neighbor_ratio": 0.1,
            "forgotten_ratio": 0.1,
            "buffer_k": 10,
            "holdout_n": 5,
            "holdout_seed": 456,
        },
        "output": {"run_name": "v3h_proto_opsd_instrdialog_seed456"},
    }


class RunGuardTests(unittest.TestCase):
    def test_skip_match_and_refuse_incomplete_or_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp)
            cfg = _cfg()
            run = results / "runs" / cfg["output"]["run_name"]
            run.mkdir(parents=True)
            with self.assertRaises(SystemExit) as incomplete:
                refuse_or_skip_existing_run(str(results), cfg["output"]["run_name"], cfg)
            self.assertEqual(incomplete.exception.code, 4)

            (run / "final_metrics.json").write_text("{}", encoding="utf-8")
            import yaml

            (run / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            with self.assertRaises(SystemExit) as skip:
                refuse_or_skip_existing_run(str(results), cfg["output"]["run_name"], cfg)
            self.assertEqual(skip.exception.code, 0)

            other = _cfg(seed=789)
            self.assertNotEqual(fingerprint(cfg), fingerprint(other))
            self.assertFalse(fingerprints_match(cfg, other))
            with self.assertRaises(SystemExit) as mismatch:
                refuse_or_skip_existing_run(str(results), cfg["output"]["run_name"], other)
            self.assertEqual(mismatch.exception.code, 3)
            self.assertTrue((run / "final_metrics.json").is_file())
            _ = json.loads((run / "final_metrics.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
