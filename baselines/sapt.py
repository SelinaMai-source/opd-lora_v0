"""Existing SAPT class.

The trainer stays in ``baselines/sapt/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.sapt.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "sapt")]

from baselines.sapt.method import SAPTMethod

BASELINE_NAME = "sapt"


def build(cfg):
    return SAPTMethod(cfg)
