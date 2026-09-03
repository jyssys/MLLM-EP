"""Install read-only timing/output hooks in every vLLM CUDA worker."""
from poc_flashvep.visual_streaming_prefill_poc.hooks.streaming_hook import install

install()

if __import__("os").environ.get("FLASHVEP_REAL_STREAMING") == "1":
    from poc_flashvep.visual_streaming_prefill_poc.hooks.real_streaming_hook import install as install_real
    install_real()
