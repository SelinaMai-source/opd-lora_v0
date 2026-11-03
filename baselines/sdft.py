"""Existing SDFT class.

The trainer stays in ``baselines/sdft/method.py``. This file shares that
package name, so submodule lookup is pointed at the directory. Importing
``baselines.sdft.method`` still loads the existing class.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent / "sdft")]

from baselines.sdft.method import SDFTMethod

BASELINE_NAME = "sdft"


def build(cfg):
    return SDFTMethod(cfg)
