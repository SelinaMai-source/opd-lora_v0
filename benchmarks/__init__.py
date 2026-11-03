"""Benchmark loaders. They call ``core.data`` and do not copy the json files."""

from benchmarks.instrdialog import get_metadata as instrdialog_metadata
from benchmarks.instrdialog import get_stream as instrdialog_stream
from benchmarks.instrdialog import load_benchmark as load_instrdialog
from benchmarks.instrdialog import validate_dataset as validate_instrdialog
from benchmarks.instrdialogpp import get_metadata as instrdialogpp_metadata
from benchmarks.instrdialogpp import get_stream as instrdialogpp_stream
from benchmarks.instrdialogpp import load_benchmark as load_instrdialogpp
from benchmarks.instrdialogpp import validate_dataset as validate_instrdialogpp

__all__ = [
    "instrdialog_metadata",
    "instrdialog_stream",
    "load_instrdialog",
    "validate_instrdialog",
    "instrdialogpp_metadata",
    "instrdialogpp_stream",
    "load_instrdialogpp",
    "validate_instrdialogpp",
]
