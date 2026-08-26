"""Opt-in startup hook for live-wavefront forensics."""

import os


if os.environ.get("FLASHVEP_WAVEFRONT_FORENSICS_ENABLE") == "1":
    from poc_flashvep.live_wavefront_forensics.instrumentation import install

    install()
