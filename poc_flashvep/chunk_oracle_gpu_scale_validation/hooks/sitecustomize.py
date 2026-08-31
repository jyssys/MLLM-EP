"""Startup hook for the bounded chunk-oracle GPU replay only."""
import os

if os.environ.get("FLASHVEP_CHUNK_REPLAY_DIR"):
    from poc_flashvep.chunk_oracle_gpu_scale_validation.replay import install
    install()
