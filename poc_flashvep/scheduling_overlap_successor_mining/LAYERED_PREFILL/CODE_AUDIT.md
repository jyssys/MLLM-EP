# Layered Prefill source audit

Official `scale-snu/layered-prefill` revision
`053f80e5201a7c0ab56e468e3d578907a2ca9cc3`. Paths below are relative to that checkout.

## Execution contract, not just flags

- `nanovllm/engine/model_runner.py`: each worker selects a local CUDA ordinal;
  world size is tensor parallelism. Qwen3 dense, Qwen3 MoE and GPT-OSS are handled.
  This is not an EP all-to-all engine and has no native Qwen3-VL image encoder,
  MRoPE or DeepStack path. Transfer claims must distinguish a faithful trace oracle
  from an actual end-to-end VL port.
- `nanovllm/models/qwen3_moe.py`: every TP rank retains all logical experts with
  tensor-sharded expert matrices and output reduction. Do not label TP2 as EP2.
- `nanovllm/engine/scheduler.py`: decode has priority. A layered prefill cohort
  advances one layer group per iteration while decode advances all layers.
  Active prefill cohorts share their stage count; admission cannot freely mix
  different partially advanced cohorts. This lifecycle constraint is separate
  from the simple number-of-groups knob.
- `get_num_stages(T, attn_workload)` already changes groups with input length:
  thresholds 512/1024/2048/4096 map to 1/2/4/8 (or 6) groups, then configured
  count. `attn_workload` is not used in the decision. Supported queue mappings
  include configured counts 4/12/16/24; CLI default 2 must not be blindly used.
- Group membership in `model_runner.prepare` is contiguous equal-layer
  `np.array_split`. Intermediate hidden/residual and positional tensors persist
  across serving iterations. A cost-balanced oracle must charge this state and
  preserve dependencies, not simply divide summed layer costs.

## Reproduction and measurement hazards

- Official Qwen commands: TP2, chunked prefill budget 512 vs layered budget 8192,
  configured groups 16. Primary precision must explicitly be BF16; a float16
  default is not equivalent to the paper's BF16 experiment.
- Required flash-attention fork is `d9e577e1b9c62f0d92d0a263d792de41a554e0bd`,
  plus the repository's `flash-attention.patch`; torch 2.8.0/CUDA 12.8.
- Shared-memory segment is named `nanovllm`: run only one engine from this
  baseline at a time, and never clean unrelated shared memory.
- Eager disables model CUDA graphs but does not necessarily disable compiled
  attention helpers. Warmup and independent restarts remain required.
- Expert reload accounting is a bytes/work proxy until hardware traffic is
  actually measured. It is not a DRAM bandwidth counter.
- API endpoint/client streaming must be verified against native token-ready
  timestamps, not inferred from batched HTTP delivery.

## Candidate assumption and falsifiers

The defensible assumption is not 'group count is static'. It is that
length-adaptive, contiguous approximately equal-layer cohorts remain adequate
when layer cost and prefill/decode composition change. First attack with existing
group count and chunk knobs. If those recover at least 80% of any request-level
gap, classify incremental. No material failure is established yet.

## Reproduction harness staging

The first startup/correctness probe uses eager execution to localize failures.
`run_configuration.py --cuda-graph` restores the official default graph-enabled
mode; a successful eager smoke is not evidence of paper-equivalent performance.
Both scheduling policies must use the same graph mode and pass output checks.
Optional measurement patch carries client request identity and installs CUDA
events after startup warmup; clean request runs keep the hook disabled.
