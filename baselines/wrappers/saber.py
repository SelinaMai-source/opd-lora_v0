"""Existing SABER class."""
from baselines.saber.method import SABERMethod

BASELINE_NAME = "saber"


def build(cfg):
    return SABERMethod(cfg)
