"""InstrDialog++ stream. Loading is delegated to ``core.data``."""
from __future__ import annotations

from typing import Any, Dict, Optional

from benchmarks._common import load_stream, metadata, resolve_stream_file, validate_dataset
from core.data import ContinualStream

DIRNAME = "instrdialogpp"
FILENAME = "citb_cl_38_random_tasks_train50_eval10_multiref.json"
CITATION = (
    "CITB InstrDialog++, Zhang et al., Findings of EMNLP 2023, arXiv:2310.14510; "
    "tasks come from Super-NaturalInstructions, arXiv:2204.07705"
)


def get_metadata() -> Dict[str, str]:
    return metadata("InstrDialog++", DIRNAME, FILENAME, CITATION)


def load_benchmark(root: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> ContinualStream:
    return load_stream(resolve_stream_file(DIRNAME, FILENAME, root=root, config=config))


def get_stream(root: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> ContinualStream:
    return load_benchmark(root=root, config=config)


__all__ = ["get_metadata", "load_benchmark", "get_stream", "validate_dataset"]
