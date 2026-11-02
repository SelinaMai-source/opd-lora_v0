#!/usr/bin/env python3
"""Write v3c yaml files. Does not touch v3h or older configs."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from core.replay_budget import validate_v3c_replay_block  # noqa: E402
from v3c_matrix import all_runs, build_config, expected_counts  # noqa: E402


def main() -> None:
    p1, p2, p3, total = expected_counts()
    if (p1, p2, p3, total) != (12, 16, 8, 36):
        raise SystemExit(f"matrix size changed: {(p1, p2, p3, total)}")
    written = []
    for name in all_runs():
        cfg = build_config(name)
        validate_v3c_replay_block(cfg["replay"])
        if int(cfg["seed"]) not in {456, 789, 42}:
            raise SystemExit(f"unexpected seed in {name}")
        if int(cfg["replay"]["holdout_seed"]) != 456:
            raise SystemExit(f"holdout_seed drifted in {name}")
        dest = ROOT / "configs" / f"{name}.yaml"
        dest.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        written.append(dest.name)
    print(f"wrote {len(written)} v3c configs")


if __name__ == "__main__":
    main()
