"""Existing CITB L2-anchor class."""
from baselines.citb.l2.method import CITBL2Method

BASELINE_NAME = "citb_l2"


def build(cfg):
    return CITBL2Method(cfg)
