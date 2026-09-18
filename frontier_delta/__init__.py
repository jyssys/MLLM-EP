"""Dense semantic emulation and exact-codec probes for FrontierEP/DeltaEP."""

from .emulator import (
    FrontierConfig,
    FrontierGenerationResult,
    decode_word_bitmap,
    encode_word_bitmap,
    generate_frontier,
    save_frontier_trace,
)

__all__ = [
    "FrontierConfig",
    "FrontierGenerationResult",
    "decode_word_bitmap",
    "encode_word_bitmap",
    "generate_frontier",
    "save_frontier_trace",
]
