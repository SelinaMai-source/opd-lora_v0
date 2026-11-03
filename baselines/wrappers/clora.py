"""Existing CLoRA class."""
from baselines.clora.method import CLoRAMethod

BASELINE_NAME = "clora"


def build(cfg):
    return CLoRAMethod(cfg)
