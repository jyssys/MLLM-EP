"""Opt-in fusion-horizon causal hook."""

import os

if os.environ.get("FLASHVEP_FUSION_CONTROL"):
    from poc_flashvep.cross_image_fusion_horizon.instrumentation import install
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe

    install_backend_probe()
    install()
