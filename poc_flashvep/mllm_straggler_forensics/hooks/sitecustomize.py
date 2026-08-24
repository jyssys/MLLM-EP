"""Opt-in startup hook for targeted expert replay/profiling."""

from poc_flashvep.mllm_straggler_forensics.instrumentation import install

install()
