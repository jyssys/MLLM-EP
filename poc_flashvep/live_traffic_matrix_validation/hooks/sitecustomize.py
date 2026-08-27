"""Opt-in worker hook for live source→destination DeepEP tracing."""

import os

if os.environ.get("FLASHVEP_MATRIX_ENABLE") == "1":
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.live_traffic_matrix_validation.instrumentation import install

    install_backend_probe()
    install()
