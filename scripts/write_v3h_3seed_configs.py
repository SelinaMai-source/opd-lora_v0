#!/usr/bin/env python3
"""Copy the 8 verified seed-456 v3h yamls to seed 789/42.

Only `seed` and `output.run_name` change. `holdout_seed` stays 456.
Do not use scripts/write_v3_configs.py for this matrix: it writes
holdout_seed = training seed.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path("/root/research-repos/opd-lora_v3")
OUT = ROOT / "configs"
METHODS = ("proto_ce", "proto_opsd", "rand_ce", "rand_opsd")
STREAMS = ("instrdialog", "instrdialogpp")
SEEDS = (789, 42)


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def main() -> None:
    written = []
    for method in METHODS:
        for stream in STREAMS:
            src = OUT / f"v3h_{method}_{stream}_seed456.yaml"
            if not src.is_file():
                raise FileNotFoundError(src)
            raw = src.read_text(encoding="utf-8")
            cfg = yaml.safe_load(raw)
            replay = cfg.get("replay") or {}
            if int(replay.get("holdout_seed", -1)) != 456:
                raise SystemExit(f"{src} holdout_seed is {replay.get('holdout_seed')}, expected 456")
            if str(replay.get("monitor")) != "train_holdout":
                raise SystemExit(f"{src} monitor is {replay.get('monitor')}, expected train_holdout")
            for seed in SEEDS:
                name = f"v3h_{method}_{stream}_seed{seed}"
                new_cfg = yaml.safe_load(raw)
                new_cfg["seed"] = int(seed)
                new_cfg["output"]["run_name"] = name
                if int(new_cfg["replay"]["holdout_seed"]) != 456:
                    raise SystemExit(f"{name} holdout_seed mutated")
                dest = OUT / f"{name}.yaml"
                dest.write_text(dump(new_cfg), encoding="utf-8")
                written.append(str(dest))
                print(f"wrote {dest} seed={seed} holdout_seed={new_cfg['replay']['holdout_seed']}")
    print(f"total={len(written)}")


if __name__ == "__main__":
    main()
