"""Existing SAPT class."""
from baselines.sapt.method import SAPTMethod

BASELINE_NAME = "sapt"


def build(cfg):
    return SAPTMethod(cfg)
