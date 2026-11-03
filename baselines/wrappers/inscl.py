"""Existing InsCL class."""
from baselines.inscl.method import InsCLMethod

BASELINE_NAME = "inscl"


def build(cfg):
    return InsCLMethod(cfg)
