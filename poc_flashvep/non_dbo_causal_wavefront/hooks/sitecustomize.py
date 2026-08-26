"""Opt-in startup hook for the non-DBO stage-wavefront PoC."""

import os


if os.environ.get("FLASHVEP_NON_DBO_WAVEFRONT_ENABLE") == "1":
    from poc_flashvep.non_dbo_causal_wavefront.instrumentation import install

    install()
