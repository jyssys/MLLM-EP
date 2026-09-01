"""Install ASAP instrumentation in vLLM's spawned GPU worker interpreters."""
import os

if os.environ.get("FLASHVEP_SERVING_PROBE") == "1":
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.ep4_serving_straggler_regime.serving_probe import install as install_scheduler
    install_backend_probe(); install_scheduler()
    if os.environ.get("FLASHVEP_ASAP_BASE_ONLY") == "1":
        from poc_flashvep.ep4_serving_straggler_regime.live_instrumentation import install
    else:
        from poc_flashvep.asap_sync_phenomenon_reproduction.asap_instrumentation import install
    install()
