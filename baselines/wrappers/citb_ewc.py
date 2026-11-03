"""Existing CITB EWC class."""
from baselines.citb.ewc.method import CITBEWCMethod

BASELINE_NAME = "citb_ewc"


def build(cfg):
    return CITBEWCMethod(cfg)
