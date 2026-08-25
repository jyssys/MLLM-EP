"""Opt-in functional expert-output capture hook."""

import os

if os.environ.get("FLASHVEP_FUNCTIONAL_CONTROL"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.visual_expert_functional_redundancy.instrumentation import install

    install_backend_probe()
    install()
