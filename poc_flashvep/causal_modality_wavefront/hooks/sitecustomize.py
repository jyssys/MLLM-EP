"""Opt-in hook for scheduler-free causal-modality replay."""

import os

if os.environ.get("FLASHVEP_CAUSAL_WAVEFRONT_ENABLE") == "1":
    from poc_flashvep.causal_modality_wavefront.operator_replay import install

    install()
