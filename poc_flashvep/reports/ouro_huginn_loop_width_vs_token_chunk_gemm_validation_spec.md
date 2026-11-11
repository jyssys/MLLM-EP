# Loop width versus Token chunk: GEMM and serving validation

## Research question

Determine whether width-preserving Loop splitting retains an advantage after strong shape-specific GEMM tuning.
Do not assume that a positive result must exist.
Find reproducible operating conditions, not a universal dominance claim.
Retain negative results and correctness failures.

Primary example: Huginn R32, 2048 input tokens.
Compare one full-prompt Loop with a 64-token suffix across all 32 Loops.
Both contain 2048 recurrent token rows, but their Attention and boundary work differ.
Use Ouro R4, 4096 input tokens, as a second-model control.

## Preserved contract

- Original pinned BF16 weights and fixed R32/R4.
- Complete Attention, KV, Hidden State, and recurrent computation.
- No approximation, quantization, early exit, or changed SLO.
- Preserve the existing staged index, active checkout, external runtime, and earlier evidence.
- Reuse Native transfer, graph, reservation, and qualification paths.
- Keep inactive requests on the qualified fast path.

Publication parent: `4470f9e2a1c6ea1c1112c59ed751fa921bcaf2dd`.
Use a new evidence directory and publication branch.

## Competing hypotheses

H1: Fixed 256-row projection padding and kernel selection explain most of the serial gap.
Prediction: qualified small-row kernels substantially reduce Token cost.

H2: Width preservation retains an advantage after kernel tuning.
Prediction: large-row work uses less total device time under matched recurrent rows or matched measured P savings.
Do not treat those two matching conditions as equivalent.

H3: Execution granularity reverses the serving preference.
Prediction: Token work fits short execution budgets that reject a full-prompt Loop.
Longer budgets may favor the more efficient complete Loop.
Microbenchmark packing is not evidence of online Decode safety.

## Phase 1: Projection-kernel experiment

Use real checkpoint weights and reproducible BF16 operands.
Test representative attention, MLP, and adapter projection geometries.
Test row counts 16, 32, 64, 128, 256, 512, 1024, 2048, and 4096 where applicable.

Compare these implementations:

1. The qualified 256-row Down/adapter implementation.
2. Unpadded library GEMM as a numerical and performance diagnostic.
3. Shape-specific cuBLASLt heuristic candidates with a common workspace cap.

Also check valid no-split-K variants and canonical-256 algorithms on the real unpadded layout.
Keep the same maximum candidate count for each shape.
Canonical algorithm reuse must not execute dummy rows.

Search and timing use separate operand seeds.
Record library versions, algorithm attributes, workspace bytes, warm-up, eager timing, and graph replay timing.
Use repeated independent GPU processes for promising configurations.
“Best” means the best validated candidate in this finite search.
It does not mean a mathematically optimal GEMM implementation.

Check exact BF16 output equality against the qualified projection reference.
Also record error magnitude for rejected candidates.
An operator-level pass does not qualify model state or serving.
Do not replace the qualified numerical fix with an inaccurate fast kernel.

## Phase 2: Whole recurrent-path attribution

For a qualified candidate, reuse Native serial Token and Loop continuation.
Compare the original and candidate projection implementations under the same runtime.
Separate dense projections, Attention geometry, boundary work, graph calls, and transfer accounting.
Report CUDA Event enclosures as enclosures, not exclusive kernel union.
Do not activate intrusive CUPTI during primary measurements.

Qualify N8 and then N32 before claiming exact runtime improvement.
Check output tokens, valid prompt KV, Hidden State, full work, ACK, and final resource retirement.
Stop an inaccurate candidate before performance expansion.
If no candidate qualifies, retain its microbenchmark as diagnostic-only.

## Phase 3: Workload and execution-budget frontier

Give Token splitting large suffixes and compatible-request batching where the runtime supports them exactly.
Give both families the same tuning and graph budgets.
Compare multiple cuts, not only 64-token Token work against a full Loop.
Separate per-request comparisons from cohort-level equal-work comparisons.

Primary anchors remain Huginn 2048/16 at QPS 1.971 and Ouro 4096/32 at QPS 6.356.
Discovery can also use low load, output 4, and controlled execution budgets.
Label each condition explicitly.
Use held-out arrival seeds for confirmation.

Measure useful P work saved, D occupation, largest execution quantum, handoff, and Decode interference.
Keep the frozen absolute SLO and the Native-relative 5% Decode-tail target.
Use marker-free output latencies for serving.
Do not substitute Host receipts for exact Service TTFT.

Only expand a surviving correctness and cost candidate into N128 serving.
If Token and Loop have complementary feasible regions, implement only the smallest justified selector.
Compare that selector with both best fixed choices.
Do not build a new serving engine before this gate.

## Fairness and limitations

Count ideal Attention pairs, padded dense work, and Prelude/Coda separately.
Logical work, hardware counters, modeled work, and measured time remain separate.
Kernel tuning must apply symmetrically when the same shape occurs in either family.
A graph-compatible or high-throughput implementation can still worsen Decode tails.
A generic logical-layer implementation with the same frontier is an equivalent mechanism.
Shared weights alone do not prove a unique Loop-splitting contribution.

### Batch-coverage amendment

The N32 qualification exposed additional P matrix geometries.
Examples include Huginn M3968 and Ouro M6144/M8192.
The first kernel map does not cover those shapes.
Serial single-request costs remain applicable to their measured shapes.
Final serving must not give one method an asymmetric untuned fallback.

Search the missing batch geometries with the same candidate and workspace limits.
Keep every original geometry's start-1 algorithm unchanged.
Add new geometries from the additional search's start 1 only.
Use starts 2 and 3 as confirmation.
Freeze a separate v2 map before serving.
Run fresh N8 and N32 state qualification for every serving arm.

### Serial component observation

Serial controls also record each legal recurrent graph execution with CUDA Events.
These enclosures exclude separate Prelude and prompt Coda calls.
They can still include output clones and host launch gaps.
They are not exclusive kernel times.
The observer does not execute extra recurrent work.
Marker-free serving disables this observer.

### Realized-quantum amendment

The inherited Token cap excludes the initial Prelude unless explicitly enabled.
The first serial Token controls therefore execute all 32 Loops in one submission.
Their total costs remain valid for that grouped execution.
Their nominal cap-1 and cap-32 labels do not establish different execution policies.

The v2 Token configurations apply the cap to Prelude and recurrent phases.
The final comparison includes both paced and grouped execution.
Each nonzero v2 arm must pass an observed D-quantum audit.
N8, N32, and serving fail the campaign gate if submitted quanta exceed their limits.
Keep the first alias configurations and their raw observations.
Do not use them as small-quantum Decode-protection evidence.

## Decisions

- `QUALIFIED_LOOP_WIDTH_ADVANTAGE`: a width benefit survives the tested kernel search and runtime correctness gates.
- `PADDING_OR_KERNEL_ARTIFACT`: the apparent benefit disappears after valid kernel improvement.
- `COMPLEMENTARY_REGIONS`: reproducible Token and Loop operating regions support selection.
- `SERVING_LOOP_BENEFIT`: held-out serving improves while preserving Decode protection.
- `NO_GO_NET_BENEFIT`: measured overhead or interference removes the serving benefit.
- `INCONCLUSIVE_CORRECTNESS`: faster candidates cannot preserve the required state contract.
- `INCONCLUSIVE_MEASUREMENT`: evidence cannot support the comparison.

These decisions are model-specific.
Do not label a microbenchmark-only result as serving success.

## Resource and closure policy

Start: 2026-11-11 07:38:55 UTC.
Review progress after four hours. Reserve closure time before the six-hour operational safety bound.
Each GPU job has its own timeout.
Stop only verified owned burners or this experiment's process groups.
Use isolated timing and restore approved burners when GPU work closes.
Verify all GPUs 0–7 before the final response.

Final report: `ouro_huginn_loop_width_vs_token_chunk_gemm_validation_final_summary.md`.
Publish reports, source, configurations, failures, and evidence through a private Git index.
Push a new branch and verify the Remote SHA.
