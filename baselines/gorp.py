"""Existing GORP class.

The trainer stays in ``baselines/gorp/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.gorp.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "gorp")]

from baselines.gorp.method import GORPMethod

BASELINE_NAME = "gorp"


def build(cfg):
    return GORPMethod(cfg)
