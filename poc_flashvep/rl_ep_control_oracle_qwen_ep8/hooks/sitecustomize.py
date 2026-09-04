"""Experiment-local child-worker hook.

The validated read-only CUDA-event hook is installed first.  The second hook
wraps the router only when ``FLASHVEP_ACTION`` requests TEMP_BALANCE; KEEP is
bit-for-bit route preserving.  This file is intentionally outside installed
vLLM source.
"""
from __future__ import annotations

import os

if os.environ.get("FLASHVEP_MATRIX_ENABLE") == "1":
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.live_traffic_matrix_validation.instrumentation import install

    install_backend_probe()
    install()
    from poc_flashvep.rl_ep_control_oracle_qwen_ep8.action_router import install as install_action

    install_action()
