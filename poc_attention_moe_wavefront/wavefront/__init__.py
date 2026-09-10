"""Analysis primitives for the Attention–MoE wavefront Phase-0 PoC."""

from .oracle import (  # noqa: F401
    DEFAULT_FRACTIONS,
    amdahl_adjusted_ttft,
    candidate_splits,
    compute_oracles,
    ideal_wave_ms,
)

