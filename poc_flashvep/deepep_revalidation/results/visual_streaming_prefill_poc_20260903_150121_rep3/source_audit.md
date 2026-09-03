# Visual streaming prefill source audit

## Runtime path

The measured run used the existing Qwen3-VL-30B-A3B-Instruct snapshot with
BF16, vLLM 0.20.0 V1, TP2/DP2/EP4/PP1, DeepEP high-throughput,
TritonExperts, linear expert placement, DBO off, prefix caching off, eager
execution, and `max_num_batched_tokens=8192`.  Two DP owners each ran a TP2
engine, covering only physical GPUs 1--4 through
`CUDA_VISIBLE_DEVICES=1,2,3,4`.

The hook in `poc_flashvep/visual_streaming_prefill_poc/hooks/streaming_hook.py`
is read-only.  It records CUDA-event durations around the real
`Qwen3_VisionTransformer.forward` and every
`Qwen3MoeDecoderLayer.forward`; it does not alter hidden states, routing,
expert placement, precision, or scheduling.  Rows are appended to JSONL at
the end of each call so worker teardown cannot discard the trace.

## Image independence

In the installed `vllm/model_executor/models/qwen3_vl.py`,
`encoder_eager_forward` calls `self.visual(pixel_values, grid_thw)` and the
vision transformer uses per-image `grid_thw`/`cu_seqlens` metadata.  The
multimodal path later splits the concatenated output using image sizes
(`_process_image_input`).  Thus image sequences are structurally independent
inside the encoder.  The current API returns one concatenated embedding tensor
after the complete call and exposes no per-image ready callback or LM input
injection boundary.

## Boundary implication

Independent image forwards are therefore a valid bounded equivalence test,
but image-ready -> partial LM prefill requires a runtime boundary change (a
per-image encoder work unit/callback and a safe embedding handoff).  No such
runtime change was implemented in this PoC.  DeepEP/LM execution remained the
stock path.

## Timing caveat

The hook synchronizes each CUDA event to make layer/encoder durations
observable.  Consequently driver wall time includes instrumentation and
vLLM scheduling/DP overhead and is not treated as clean TTFT.  The oracle
uses only paired CUDA-event stage measurements and refuses negative cumulative
prefill increments instead of clipping them.
