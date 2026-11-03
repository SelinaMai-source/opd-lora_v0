"""Existing SDFT class."""
from baselines.sdft.method import SDFTMethod

BASELINE_NAME = "sdft"


def build(cfg):
    return SDFTMethod(cfg)
