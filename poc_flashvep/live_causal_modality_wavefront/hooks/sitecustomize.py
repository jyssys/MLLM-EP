"""Opt-in worker hook for the live causal wavefront experiment."""

import os

if os.environ.get("FLASHVEP_LIVE_WAVEFRONT_CONTROL"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.live_causal_modality_wavefront.instrumentation import install

    install_backend_probe()
    install()
