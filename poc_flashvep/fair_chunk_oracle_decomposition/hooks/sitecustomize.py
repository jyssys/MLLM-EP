import os
if os.environ.get("FAIR_REPLAY_DIR"):
    from poc_flashvep.fair_chunk_oracle_decomposition.replay import install
    install()
