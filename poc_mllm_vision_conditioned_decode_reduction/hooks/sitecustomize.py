"""Opt-in worker-process installation for bounded decode vector capture."""

from __future__ import annotations

import os

if os.environ.get("MLLM_VCDR_CAPTURE_ENABLE") == "1":
    from poc_mllm_vision_conditioned_decode_reduction.decode_vector_capture import install

    install()
    os.environ["MLLM_VCDR_CAPTURE_HOOK_IMPORTED"] = "1"
