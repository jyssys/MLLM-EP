"""Opt-in composition of the validated DBO wavefront hook and stage timing."""

import os

if os.environ.get("FLASHVEP_LIVE_WAVEFRONT_CONTROL"):
    from poc_flashvep.deepep_revalidation.backend_probe import install_backend_probe
    from poc_flashvep.live_causal_modality_wavefront.instrumentation import install as install_wavefront
    from poc_attention_moe_wavefront.naive_addon import install as install_stage_timing

    install_backend_probe()
    install_wavefront()
    install_stage_timing()

