"""Install only this experiment's read-only probes in vLLM worker interpreters."""

import os

if os.environ.get("FLASHVEP_SERVING_PROBE") == "1":
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.dp_ep_arrival_skew_two_topologies.arrival_instrumentation import install
    from poc_flashvep.ep4_serving_straggler_regime.serving_probe import install as install_scheduler

    install_backend_probe()
    install()
    install_scheduler()
