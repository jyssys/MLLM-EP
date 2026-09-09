# NanoFlow source audit

Three official source lines were inspected rather than assuming the default
branch is the OSDI artifact:

| Source | Pin | Evidence |
|---|---|---|
| default Nanoflow-python | f179a907828b87e042126e585880301ff5b2c62a | Python operation graph; Llama paths; H100 CMake |
| original main | d6b381e58110a8b5d08cfabd4a55c0d5d0ebef57 | C++ pipeline; Qwen2-72B native config; A100-oriented build |
| official dev-h100 | 915790ea862d1ddd52a8871282c1eb5be88f1391 | Packaged Python graph plus Qwen1.5/Qwen2 MoE TP/EP paths |

## Important corrected discovery

The default branch and original main did not expose a runnable Mixtral model
implementation where expected (JSON configurations alone are insufficient).
However `dev-h100` contains actual Qwen MoE implementations and native multi-GPU
entrypoints. Therefore 'NanoFlow has no MoE code' would be an incorrect conclusion.
Prefer this official H100 path before a full Qwen3-VL port.

## Native Qwen path

- `nanoflow/entry/common.py`: Qwen1.5-MoE-A2.7B-EP; Qwen2-57B-A14B-EP;
  Qwen2-57B-A14B-TP-EP. Last path asserts world=TP=EP. Configurable nano splitting,
  CUDA graphs, double buffering and automatic search are independent flags.
- `models/qwen2_moe*/`: real shared and routed expert branches; replicated
  hidden inputs with expert-index partitioning and output collectives. Must not
  call this DeepEP token-dispatch unless actual communication code proves it.
- `operations/fused_moe/fused_moe.py`: router softmax/topk, optional renormalization;
  reference torch expert loop or FlashInfer `cutlass_fused_moe` with EP size/rank.
  Expert choice cannot be replaced by synthetic routes for model correctness.
- `core/basePipeline.py` and `operations/operation_base.py` own streams and graph
  dependencies. Green contexts partition SM resources; nvmath selects BLAS SM
  targets. These mechanisms need live verification, not only enabled flags.
- `core/bufferAllocate.py` and operation/search modules import Gurobi. Restricted
  license limits may affect full optimization; a solver error is not plan regret.
- `entry/common.py` hard-codes `/code/hf` checkpoint paths and some saved plans.
  Local path adaptation is permissible, with exact diffs and pinned checkpoints.

## Bounded isolated setup

Official scripts modify global packages, sysctls, pybind headers and `/usr/local`,
and invoke very large compile parallelism. Do not run them wholesale on the shared
host. Use an isolated Python environment, local prefix, bounded compiler jobs.
Required submodule pins include FlashInfer d3e9b4402a05b658bcd837250a205fd2ef7ce1bd,
MSCCL++ cdaf3aea3d767ba65dd3b08984d76bd50615f92e.

Official Meta Llama3-8B checkpoint access returned HTTP401/GatedRepoError.
No gated mirror workaround is used. Ungated official Qwen MoE is a legitimate
supported baseline alternative, conditional on correctness and kernel API sanity.

## Transfer gate

Compare one static plan, per-workload static, a small portfolio and per-regime
oracle. Charge profiling/search/reconfiguration and request critical paths.
Known paper low-load inefficiency or an unsupported port is not a new failure.
No material failure is established yet.

## 2026-09-09 native search-path boundary

`BasePipeline.init_streams` creates green-context streams for 8..120 SM in
increments of eight, plus a full-132-SM stream; `config_streams` consumes each
operation plan's `p_value`. A bounded 112-SM compute / 16-SM collective control
uses this existing native option. Each category's stream pool is created by a
separate split call, so the two requested counts do **not** establish physically
disjoint cross-category SM sets. Runtime active-plan stream IDs/counts are saved;
do not call the resource control an optimized or proven-disjoint partition.

The live Qwen EP factory constructs `BasePipeline` with only COMP and MEM
categories and does not override `init_category`. The bounded adapter assigns
the routed-output all-reduce to the second (MEM-labelled) stream and other FFN
operations to COMP. These labels select actual streams; this is not a claim that
NCCL is a memory operator in the paper's resource model. The paper's dense
automatic-search model instead tags collectives NET.

`BasePipeline.init_streams` creates green-context SM partitions (8 through 120)
and a 132-SM full-device stream per configured category. A plan's `p_value`
therefore really changes the stream's allowed SM set; inspecting only the
FlashInfer call's keyword arguments would miss this mechanism. The current
manual diagnostic uses 132, so it has NOT searched execution-unit allocation.

The pinned `FusedMoE` class lacks `init_profile_db`, `store_profile_db`,
`profile_update`, and `profile_run` implementations. The inherited database
methods raise `NotImplementedError`; the inherited execution hook is empty.
The supplied automatic-search entrypoint instantiates dense Llama3-70B and reads
its per-operation profile tables. Consequently, running that entrypoint against
Qwen without faithful MoE profiling is not an official searched-optimum baseline.
Empty-hook timings must never be entered as zero-cost expert profiles.

These are port/search coverage limitations, NOT measured method failures.
Native splitter/executor, fixed-plan numerical checks and CUDA overlap traces
remain useful bounded diagnostics. A finite manual-plan envelope only bounds
choices in that portfolio, not every schedule the paper could search.

## Prefill output materialization scope

The Qwen EP factory's `GetLogits`, `ModelLayerNorm`, and `Sampling` use
`last_only()`, but the base implementation sets **last layer**, not last token.
`BasePipeline.run` creates `global_batch_size` output slots and only afterwards
selects `cumsum_input[1:] - 1` positions on the CPU. Thus prompt-wide final-head
materialization is a source-supported compatibility/performance concern for
large prefill in this H100 path. New numerical captures record the actual
native logits shape and selected token positions to verify this inference.
No request-time saving is attributed without a measured control. Last-token
logit selection is an obvious existing engineering optimization, not a novel
NanoFlow successor, and this path must not be mistaken for the fully optimized
OSDI paper prefill implementation.
