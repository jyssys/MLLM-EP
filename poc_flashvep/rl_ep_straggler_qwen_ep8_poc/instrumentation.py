"""Reuse the validated read-only Qwen3 MoE CUDA-event hook.

The hook records local expert assignment histograms and dispatch/expert/
combine CUDA-event spans. It does not alter routes, weights, placement, or
the scheduler.
"""

from poc_flashvep.live_traffic_matrix_validation.instrumentation import install

__all__ = ["install"]
