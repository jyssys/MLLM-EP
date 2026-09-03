# Visual streaming feasibility

**POSSIBLE_WITH_RUNTIME_CHANGE**. Qwen3-VL concatenates image patch sequences with `grid_thw`/`cu_seqlens` and runs the vision transformer before merging embeddings into the language sequence. Image sequences are structurally separated, but the current vLLM path returns the complete visual embedding tensor before LM execution and has no per-image ready callback. Exposing image-level completion would require runtime scheduling/interface work; no implementation was added.
