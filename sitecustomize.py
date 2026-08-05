"""Project-local Python startup hooks.

This file is intentionally inert unless an explicit project hook environment
variable is set. When enabled, vLLM worker subprocesses inherit the environment
and install the requested patch before model construction.
"""

from __future__ import annotations

import os
from importlib.util import find_spec


if os.environ.get("VLLM_MOE_EXPERT_MAP_JSON"):
    from vllm_custom_placement import apply_vllm_custom_placement_patch

    apply_vllm_custom_placement_patch()

if os.environ.get("FLASHVEP_PROFILE_JSONL") and find_spec("torch") is not None:
    from poc_flashvep.flashvep.instrumentation import install

    install()

if (
    os.environ.get("FLASHVEP_PHASE1B_AUDIT_JSONL")
    or os.environ.get("FLASHVEP_PHASE1B_PROFILE_JSONL")
) and find_spec("torch") is not None:
    from poc_flashvep.flashvep.instrumentation_phase1b import install_phase1b

    install_phase1b()

if os.environ.get("FLASHVEP_OFFLINE_RESULT_DIR") and find_spec("torch") is not None:
    from poc_flashvep.offline_wavefront.offline_moe_runner import (
        install_offline_wavefront,
    )

    install_offline_wavefront()
