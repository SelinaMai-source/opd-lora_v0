"""Existing sequential LoRA class. No second implementation."""
from baselines.basic_baselines.sequential_lora.method import SequentialLoRAMethod

BASELINE_NAME = "sequential_lora"


def build(cfg):
    return SequentialLoRAMethod(cfg)
