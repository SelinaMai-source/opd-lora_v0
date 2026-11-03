"""Existing CITB A-GEM class."""
from baselines.citb.agem.method import CITBAGEMMethod

BASELINE_NAME = "citb_agem"


def build(cfg):
    return CITBAGEMMethod(cfg)
