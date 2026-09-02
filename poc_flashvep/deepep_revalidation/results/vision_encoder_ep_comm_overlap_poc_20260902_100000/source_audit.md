# Source and reference audit

## Runtime

The run used the existing `/home/esjung/.venvs/flashvep-deepep-v020`
launcher (Python 3.12.13, PyTorch 2.11.0+cu129, CUDA 12.9, vLLM 0.20.0,
Triton 3.6.0) with `CUDA_VISIBLE_DEVICES=1,2,3,4`.  vLLM reported
`DeepEPHTPrepareAndFinalize`, the linear EP mapping (32 of 128 experts per
rank), and the unquantized Triton expert backend.  No source under the
installed vLLM/DeepEP trees was modified.

## Qwen3-VL execution path

In the installed `vllm/model_executor/models/qwen3_vl.py`,
`Qwen3_VisionTransformer` (class near line 519, `forward` near line 784)
executes patch embedding, positional metadata, and a loop over vision blocks.
`encoder_eager_forward` calls `self.visual(pixel_values, grid_thw)` (around
line 2056); image processing enters through `_process_image_input` and
`embed_multimodal`.  The hook captured a real image-derived input to one
`Qwen3_VisionBlock` and reran that same block on a separate CUDA stream.  The
captured activation was `[1024, 1, 1152]` BF16 from the live vision path.

The language decoder is supplied by `qwen3_vl_moe.py` through the Qwen3 MoE
model.  `Qwen3MoeDecoderLayer.forward` in `qwen3_moe.py` performs self-attention,
post-attention layer normalization, then `self.mlp(hidden_states)` (lines
416--436).  The hook runs after the live model reaches a real MoE call; it does
not change router IDs, weights, expert placement, or model outputs.

## DeepEP path and stream semantics

The installed `deepep_ht.py` calls `get_dispatch_layout` and `buffer.dispatch`
after obtaining the previous compute event, then switches between compute and
communication streams.  Finalization calls `buffer.combine` with the previous
event and returns an asynchronous receiver that waits on the DeepEP event
before copying the combined output.  The DeepEP C++ implementation launches
communication kernels on its internal `comm_stream`; its caller must be a
different CUDA stream (same-stream calls trigger the `event.hpp` assertion).
The PoC therefore invokes DeepEP from the default worker stream while timing
the actual internal communication stream, and waits on the returned
`EventOverlap` before each cleanup/paired barrier.

The measured replay uses an existing exact layer-24 capture (799 tokens,
top-k 8, 128 experts, EP4) and actual loaded Triton/DeepEP buffers.  Each
communication phase has one real dispatch or combine collective; the extra
combine used after dispatch-only trials is cleanup and is outside its timed
interval.  No new collective is introduced into the model's own forward.

## Related work boundary

RESONATOR characterizes compute-bound vision encoders and resource sharing
(SM/HBM-aware intra-GPU sharing and encoder DP/TP choices); no official public
RESONATOR source repository was found in the bounded search, so the status is
`NOT_PUBLICLY_FOUND`.  SpaceServe is a public earlier system that separates
encoder/decoder workers and uses CUDA MPS/libsmctrl-style resource control; the
checked read-only commit is recorded in `reference_manifest.json`.  Flux/COMET
studies fine-grained MoE communication/computation overlap and provides useful
stream/event measurement precedent; it is not used as this PoC's backend.  The
candidate here is narrower: a pending real Qwen3-VL vision-block invocation
paired with a current-request DeepEP dispatch or combine.  This experiment
does not claim overlap itself as novel.

## Nsight status

`nsys` 2024.6.2.225-246235244400v0 is installed.  A one-request smoke capture
was attempted, but multiprocessing/NCCL worker startup failed under the nsys
launcher (`gloo ... Connection closed by peer`); the CUDA-event run is the
valid primary measurement.  `ncu` is not installed.  The failure log is kept
in the result directory as `nsys_attempt.log` and is not used as a performance
result.
