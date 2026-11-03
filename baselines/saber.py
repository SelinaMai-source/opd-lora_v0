"""Existing SABER class.

The trainer stays in ``baselines/saber/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.saber.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "saber")]

from baselines.saber.method import SABERMethod

BASELINE_NAME = "saber"


def build(cfg):
    return SABERMethod(cfg)
