#!/usr/bin/env python3
"""Second entry point for the same trainer.

``python scripts/train.py --config <yaml>`` calls ``core.trainer.main``.
``python -m core.trainer --config <yaml>`` is the other entry. Neither copies
the training loop. The repo root is added so a checkout can run without a
private PYTHONPATH.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trainer import main  # noqa: E402


if __name__ == "__main__":
    main()
