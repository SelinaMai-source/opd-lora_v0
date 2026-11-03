"""Existing KPIG class."""
from baselines.kpig.method import KPIGMethod

BASELINE_NAME = "kpig"


def build(cfg):
    return KPIGMethod(cfg)
