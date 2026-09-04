import os

if os.environ.get("FLASHVEP_GRANULARITY_RESULT_DIR"):
    from poc_flashvep.modality_aware_moe_granularity.replay import install
    install()
