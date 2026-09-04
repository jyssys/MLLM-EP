# T+8h checkpoint (recorded during current execution pass)

The generic Qwen3-30B-A3B control reproduces shape sensitivity but not a
stable MLLM-specific effect.  A32 extension repeats are inconsistent; this is
now treated as first-use/state confounding (H16/H23), not a positive method
signal.  No optimization method is implemented.
