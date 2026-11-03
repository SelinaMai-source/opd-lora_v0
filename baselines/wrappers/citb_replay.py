"""Existing CITB replay class. Budget is K, not the v3c mixed-batch count."""
from baselines.citb.replay.method import CITBReplayMethod

BASELINE_NAME = "citb_replay"


def build(cfg):
    return CITBReplayMethod(cfg)
