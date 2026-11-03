"""Existing KPIG class.

The trainer stays in ``baselines/kpig/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.kpig.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "kpig")]

from baselines.kpig.method import KPIGMethod

BASELINE_NAME = "kpig"


def build(cfg):
    return KPIGMethod(cfg)
