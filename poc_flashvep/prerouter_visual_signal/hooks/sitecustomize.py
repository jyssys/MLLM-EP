"""Opt-in worker hooks for direct router and vision-encoder capture."""

import os

if os.environ.get("FLASHVEP_PREROUTER_CONTROL"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.prerouter_visual_signal.instrumentation import install

    install_backend_probe()
    install()
