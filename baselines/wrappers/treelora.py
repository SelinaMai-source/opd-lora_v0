"""Existing TreeLoRA class."""
from baselines.treelora.method import TreeLoRAMethod

BASELINE_NAME = "treelora"


def build(cfg):
    return TreeLoRAMethod(cfg)
