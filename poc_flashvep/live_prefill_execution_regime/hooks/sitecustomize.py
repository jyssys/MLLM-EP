"""Opt-in worker hook for live Qwen3-VL expert timing."""

import os

if os.environ.get("FLASHVEP_LIVE_CONTROL"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.live_prefill_execution_regime.instrumentation import install

    install_backend_probe()
    install()
