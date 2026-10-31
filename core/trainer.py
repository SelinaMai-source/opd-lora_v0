"""Training loop entry.

The implementation is ``core.train``. This module is the supported command:
``python -m core.trainer --config <yaml>``.
"""
from __future__ import annotations

from core.train import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
