"""Existing InsCL class.

The trainer stays in ``baselines/inscl/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.inscl.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "inscl")]

from baselines.inscl.method import InsCLMethod

BASELINE_NAME = "inscl"


def build(cfg):
    return InsCLMethod(cfg)
