# Source discovery pass 2: DP coordination and materialization

Read independently of a target optimization.

* `vllm/v1/worker/dp_utils.py:99-159` performs an all-reduce of token counts,
  microbatch choice, padded counts and cudagraph mode on every DP>1 step.
  Padding is only applied when a synchronized graph/ubatch mode requires it.
* `vllm/model_executor/layers/fused_moe/modular_kernel.py:97-115` creates a
  CPU tensor from a Python expert-count list and copies it to the GPU for every
  receiver.  The source itself marks recomputing counts on GPU as a TODO.
* `gpu_worker.py:899` and the executor RPC path expose explicit dummy-batch
  participation when no useful tokens are scheduled.
* `all2all.py:197-260` fixes DeepEP HT buffer ownership and caps communication
  SM configuration at the buffer's construction value.

**Assumptions generated:** A08--A10, A15--A16, A20--A24, A28--A30, A39--A42.

The prior controls show why these deserve analysis but not automatic promotion:
CPU/NCCL DP switch was -0.03% request median, dummy participation -0.26%, and
the metadata call was typically 0.1137 ms.  Structural ownership is a useful
candidate source, not evidence of a paper-sized opportunity.
