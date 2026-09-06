# Source discovery pass 2 — scheduler, DP rendezvous, and metadata

Date: 2026-09-07. This pass read scheduler/runner/prepare-finalize code without
starting from a desired result.

## Paths read

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/dp_utils.py`
- `vllm/config/vllm.py`
- `vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py`
- `vllm/model_executor/layers/fused_moe/modular_kernel.py`

## Unexpected implementation facts

1. Asynchronous scheduling is auto-enabled for this executor. For MoE DP>1,
   vLLM then auto-selects CPU rather than NCCL for DP descriptor
   synchronization to avoid a GPU synchronization point.
2. Even in eager mode with DBO off and no DP padding, every model step still
   enters a CPU process-group all-reduce to exchange token counts and execution
   mode. The all-reduce is therefore a cross-engine phase rendezvous rather
   than merely a CUDA-graph padding helper.
3. DeepEP dispatch first returns an asynchronous receiver; the receiver waits
   for the event on the current stream. Combine similarly returns a receiver
   and then copies combined output into the requested output tensor.
4. For every MoE layer, `ExpertTokensMetadata.make_from_list` constructs a new
   CPU tensor from a Python list and launches a CPU-to-GPU copy. The source
   itself notes the GPU/CPU metadata copy as a TODO.

## Generated hypotheses

H17, H20, H31, H32, H33, H34, and H35.

These are not findings yet. Each requires fresh live perturbation, frequency
and direct request-level mass before promotion.
