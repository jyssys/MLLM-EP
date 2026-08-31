import os
if os.environ.get("TRUE_REPLAY_DIR"):
    if os.environ.get("TRUE_B_MODE") in {"cost", "validate"}:
        from poc_flashvep.true_gpu_chunk_oracle_poc.replay_stage_b import install
    else:
        from poc_flashvep.true_gpu_chunk_oracle_poc.replay import install
    install()
