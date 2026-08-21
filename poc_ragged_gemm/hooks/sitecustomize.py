"""Worker-side opt-in hook for the ragged-GEMM live capture."""
import os

if os.environ.get("RAGGED_GEMM_CONTROL"):
    from poc_ragged_gemm.instrumentation import install
    install()
