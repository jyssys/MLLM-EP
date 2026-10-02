def is_native_prefill(dp_tokens, prompt_tokens):
    """Existing CPU DP coordination metadata, not a new GPU-side reduction."""
    return bool(dp_tokens) and max(dp_tokens) == prompt_tokens and prompt_tokens > 1
