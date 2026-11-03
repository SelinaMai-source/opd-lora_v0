"""Existing GORP class."""
from baselines.gorp.method import GORPMethod

BASELINE_NAME = "gorp"


def build(cfg):
    return GORPMethod(cfg)
