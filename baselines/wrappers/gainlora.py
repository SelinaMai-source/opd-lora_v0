"""Existing GainLoRA class."""
from baselines.gainlora.method import GainLoRAMethod

BASELINE_NAME = "gainlora"


def build(cfg):
    return GainLoRAMethod(cfg)
