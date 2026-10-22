# Diagnostic reproduction

Read `reproduce.md` for the complete environment and burner rules.
These commands reproduce analysis, not live adaptive serving.

```bash
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.verify_surface --pattern 'p2_P*' --output main_surface_work_clock_verification.json
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.surface_analysis
/home/esjung/.venvs/ouro-chunk-loop-prefill/bin/python -m poc_flashvep.ouro_adaptive_chunk_loop.headroom
```

The same-input MLP diagnostic is registered in `p1_mlp_replay_plan.json`.
Its captured input uses the original checkpoint weights.
Its FP32 comparison is diagnostic only.
No FP32 serving result replaces BF16 generation.

Heavy controls use `measurement_control_retry_v2.json` for GPU0/GPU1.
The original failed v1 attempts must not be overwritten or silently resumed.
Other completed control workers use `measurement_control_plan.json`.
`summarize_controls` chooses the completed, reviewed attempts.

The full native CUDA traces are heavy observations.
Their wall time must not replace clean client latency.
