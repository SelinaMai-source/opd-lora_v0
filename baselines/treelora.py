"""Existing TreeLoRA class.

The trainer stays in ``baselines/treelora/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.treelora.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "treelora")]

from baselines.treelora.method import TreeLoRAMethod

BASELINE_NAME = "treelora"


def build(cfg):
    return TreeLoRAMethod(cfg)
