"""Existing GainLoRA class.

The trainer stays in ``baselines/gainlora/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.gainlora.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "gainlora")]

from baselines.gainlora.method import GainLoRAMethod

BASELINE_NAME = "gainlora"


def build(cfg):
    return GainLoRAMethod(cfg)
