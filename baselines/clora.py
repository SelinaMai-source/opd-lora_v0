"""Existing CLoRA class.

The trainer stays in ``baselines/clora/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.clora.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "clora")]

from baselines.clora.method import CLoRAMethod

BASELINE_NAME = "clora"


def build(cfg):
    return CLoRAMethod(cfg)
