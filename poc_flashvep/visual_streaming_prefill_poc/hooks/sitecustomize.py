"""Install read-only timing/output hooks in every vLLM CUDA worker."""
from poc_flashvep.visual_streaming_prefill_poc.hooks.streaming_hook import install

install()
