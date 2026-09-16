"""Trace-driven virtual Expert Parallelism simulator.

The package never creates virtual distributed ranks.  EP4/EP8 are derived
from real routing traces, measured EP2 calibration, and rank-local replay.
"""

from .mapping import ExpertOwnership, SourcePartition
from .schema import TraceBundle
from .traffic import TrafficSummary, build_traffic

__all__ = [
    "ExpertOwnership",
    "SourcePartition",
    "TraceBundle",
    "TrafficSummary",
    "build_traffic",
]
