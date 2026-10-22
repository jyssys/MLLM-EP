# Unified Ouro serving: implementation decision and plan

## Decision

**Planning verdict: CONDITIONAL.** Phase 0 is complete. Phases 1–6 are **NOT_RUN** in this study.

Approve only the native baseline qualification in Phase 1 first. Do not approve the full eight-arm implementation yet.
Static Loop PP can reuse much of native linear PP. Physical-layer PP requires cyclic stage execution, which native Ouro does not support.
That distinction makes the full program substantially larger than an Ouro model adapter.

The research value remains unproven. Static Loop PP has the same logical partition as contiguous PP over 96 tied logical layers.
A distinct contribution must come from a measured execution or KV-management benefit, not the partition name.
Async repair is a testable candidate. Its client-visible benefit remains **TO_VERIFY**.

The authoritative input is `/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_custom_vllm_unified_parallelism_planning_spec.md`.
The specification SHA256 is `e3cf38bf88a760bcd68d58f1d87e844963414692ad48aac072d210e6338b3d92`.
The audit date is 2026-10-22. No engine source, installed environment, model weight, or historical result changed.
The audit used CPU source inspection and read-only GPU inventory checks. Existing burners on GPUs 4–7 remained active.
The input specification remains outside this planning-only commit. Its server path and hash identify the audited contract.

## Evidence that changes the implementation decision

| Finding | Evidence class | Consequence |
| --- | --- | --- |
| Native Ouro uses optimized TP projections and attention | SOURCE_VERIFIED | Preserve these operators and the native TP1/TP4 path |
| Ouro lacks `SupportsPP`, ignores incoming intermediate tensors, and embeds inputs on every forward | SOURCE_VERIFIED | A PP configuration flag is insufficient |
| Native PP workers assume a forward chain and a final sampling rank | SOURCE_VERIFIED | Cyclic physical-layer PP needs a bounded reentry protocol |
| Native Ouro constructs an unused gate with parallel-input semantics | SOURCE_VERIFIED | Audit and adapt gate input sharding before Dynamic TP4 |
| Current RLT is a separate model, runner, scheduler, and cache implementation | SOURCE_VERIFIED | Port policy and lifecycle concepts, not the whole runtime |
| RLT and pinned CDB differ in sigmoid dtype and cumulative probability arithmetic | SOURCE_VERIFIED | Freeze CDB arithmetic for this program |
| Old Loop PP did not consistently beat Layer PP | HISTORICAL_MEASURED | Keep the earlier NO-GO unchanged |
| Old repair ACK precedes successor sampling | SOURCE_VERIFIED | A legal deferred wait might expose overlap, but bytes do not quantify it |

Exact source revisions and code anchors are in [evidence_index.md](evidence_index.md).

## Scope and invariant configuration

- Model: base `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
- Precision: BF16. Keep the original trained gate and model weights.
- Static: 24 shared physical layers, four loops, 96 independent logical KV namespaces.
- Dynamic: full-R4 prefill, decode q=0.5, minimum R2, maximum R4, current-row `last_exited` repair.
- GPUs: Single uses physical GPU4. Two-rank tests use GPU4,5. Four-rank tests use GPU4,5,6,7.
- Primary PP unit: a full request prompt. Do not add prompt chunking to the primary comparison.
- Native paged attention, projection kernels, sampling, frontend, and allocator remain the common foundation.
- No SGLang port, training, quantization, stale activation, or predicted exit policy.

The configured context limit is 65,536. This is a configuration fact, not a qualified runtime capacity.
Start with the previously used 8,192 runtime limit. Measure any larger limit before use.

## Development sequence and rollback boundaries

S/M/L/XL below are relative engineering estimates. They are not elapsed-time or staffing promises.
Each milestone needs a reviewed commit and an immutable result manifest before the next milestone.

### Phase 0 — This source and design audit

**Status:** PASS for planning deliverables. Runtime feasibility remains CONDITIONAL.

Reuse the historical reports and read clean upstream source snapshots. Do not install RLT into the native environment.
The complete deliverables are linked at the end of this document.

**G0:** Linear Loop PP appears compatible with bounded model and worker adapters.
Cyclic physical PP remains high risk. Require its two-rank reentry proof before promising all eight arms.
Stop that branch if it needs a replacement serving scheduler, frontend, or attention implementation.
Retain a smaller native-kernel-matched control and report the missing arm as BLOCKED.

### Phase 1 — Native Static Single and TP4 qualification

**Size:** M. **Source changes:** no model or engine behavior changes.

Create a future shared benchmark harness and read-only worker inspection hooks.
Use native `OuroForCausalLM`, `GPUModelRunner`, native sampler, and native HTTP serving.
Use a separate environment or fork. Preserve the installed historical environment.

Minimum work:

1. Freeze model, tokenizer, prompts, backend, graph policy, environment, and GPU UUID mapping.
2. Run TP1 and TP4 deterministic probes at P128/P256/P512/P2048.
3. Record logits, hidden states, KV ownership, first divergence, argmax, and actual collective participation.
4. Run 24 frozen natural requests at C1 and C4, with natural EOS and a 4096 output cap.
5. Run the 114-question historical pilot only after the bounded tests work.
6. Measure clean client timestamps and memory. Trace a separate bounded cohort.

Expected risks are TP arithmetic sensitivity, backend selection, and graph lifecycle differences.
The historical 0.03 hidden-L2 failure remains FAIL. A new same-backend contract needs independent calibration.
TP4 accuracy may run as exploratory if the numerical gate remains unresolved. It cannot silently become a qualified architecture baseline.

**G1:** Stable native generation, correct ownership, explained numerical behavior, and valid clean timing.
Unexplained suspected implementation errors block extension. Backend sensitivity alone requires explicit qualification, not a false TP failure claim.
Rollback is deletion of the new harness branch, not modification of prior results.

### Phase 2 — Static PP feasibility and matched serving

**Size:** L for linear Loop PP. **Size:** XL risk for cyclic physical PP.

Change `OuroModel` staging, local attention registration, weight loading, and `IntermediateTensors` handling.
Reuse native `MultiprocExecutor`, `Worker`, `GroupCoordinator`, paged KV allocation, and graph wrappers where their contracts hold.
Do not replace native attention with request-local PyTorch attention.

Use these pull-request-sized milestones:

| Milestone | Minimum implementation | Validation and rollback |
| --- | --- | --- |
| P2a | Pure placement and 96-namespace plan | CPU ownership tests. No serving changes |
| P2b | Linear Loop PP2: R1–2 / R3–4 | GPU4,5 logits/KV/token tests. Shared physical weights within each rank |
| P2c | Cyclic physical PP2: L1–12 / L13–24 | Correct wraparound, residual transport, ordering, cancellation, and page lifetime |
| P2d | Expand both to PP4 | Four-rank ownership and real cross-request kernel overlap |
| P2e | Tied-unrolled contiguous PP4 control | Same 96 logical boundaries, kernels, weights, graph cuts, and request policy |
| P2f | Clean C1/C4/C8 natural comparison | At least two pre-registered prompt regimes. No Dynamic implementation yet |

P2c may need worker and executor changes. Its scope must not expand silently into a new serving engine.
The scheduler counts one logical token once. It must not count each recurrent visit as another output token.

Measure graph preparation, staging, queue delay, P2P payload, residual transport, memory, and pipeline bubbles.
Include a static graph-cut control. It isolates graph fragmentation from placement effects.

**G2:** Apply the pre-registered benefit and quality gates in [acceptance_gates.md](acceptance_gates.md).
If Loop PP lacks a repeatable serving or capacity advantage, stop Dynamic PP expansion.
An equivalence to generic contiguous PP is not a new partition result.
An unimplemented cyclic arm stays BLOCKED rather than becoming an approximate substitute.

### Phase 3 — Dynamic Single, then TP4

**Size:** L. Reuse native kernels and one shared depth-policy implementation.

Expose normalized loop output and the trained gate. Add a token-local recurrent state.
Use CDB's pinned probability arithmetic. Keep full-R4 prefill and synchronous no-refill decode first.
Implement current-row repair through the native paged-cache layout, not RLT's private cache tensor.

The gate requires explicit TP handling. Native Ouro's dormant `RowParallelLinear` assumes an already partitioned input.
Use an audited input split or `input_is_parallel=False` in the Dynamic path.
Reduce partial gate outputs once. Apply bias once. Broadcast one exit decision and one survivor order across TP ranks.
Do not alter the Static path merely to activate the gate.

Add per-depth graph execution and survivor compaction. A full-R4 graph replay does not provide actual early-exit compute skipping.
Compare a static graph-cut control with Dynamic to expose added graph and host costs.

Minimum tests include mixed forced R2/R3/R4 diagnostics, page boundaries, historical-row preservation, EOS, and natural generation.
Use 24 bounded requests, then 64 GSM8K + 50 MATH pilot questions, then the frozen 400 if qualified.
Forced exits and recorded depth traces are diagnostics only.

**G3:** Correct cache state, live original gate, true skipped layer work, coherent TP collectives, and valid natural generation.
Report policy accuracy against Static separately from architecture accuracy against Single Dynamic.
No policy retuning is allowed to pass a quality gate.

### Phase 4 — Dynamic PP with synchronous repair

**Size:** L after qualified Static PP. Cyclic PP remains XL if P2c did not resolve reentry.

Reuse the same policy, admission contract, transport, and native cache accessors.
Physical-layer owners repair their own deeper namespaces locally.
Loop owners send the exit row to each skipped-depth owner.
Route coda and sampling from any exit depth without changing the original exit decision.
This needs an explicit result path. Native final-rank-only sampling is not sufficient by itself.

Minimum tests cover all exit depths, full-R4 prefill, concurrent requests, cancellation, page allocation, and natural output/depth comparison.
Measure repair wait, kernel work, queue delay, rank imbalance, and graph count separately.

**G4:** Qualify bounded Dynamic generation on supported arms before the 400-question comparison.
If synchronous repair and rank imbalance erase the Static benefit, measure required savings before proceeding.
Do not build a complex scheduler to rescue a failed placement premise.

### Phase 5 — Async repair and deferred waits

**Size:** M for an isolated transfer test. **Size:** L for safe serving integration.

First measure source packing, inter-GPU transfer, destination publication, and earlier-loop compute.
Use the exact paged-cache layout and request lifetimes from Phase 4.
Then implement only the legal overlap described in [architecture.md](architecture.md).

No-refill has two meanings that must remain separate. A strict repair-inclusive wave barrier blocks the proposed overlap.
A deferred-publication barrier may allow overlap without general slot refill. That is an explicit scheduling variant.
Compare sync and async copies within the same barrier contract. Evaluate refill separately.

Minimum validation includes delayed completion injection, cancellation, block reuse, multiple pending repairs, and graph replay buffer lifetime.
Measure receiver-local completion and the first dependent KV read. Nonblocking API calls are not evidence of overlap.

**G5:** Require real copy/compute overlap, reduced dependent wait, and repeatable client-visible improvement without new correctness failures.
If only host wait placement changes, or clean improvement stays within noise, stop.
Copy consolidation and cross-GPU aliasing remain separate ablations or future research.

### Phase 6 — Main natural-generation serving comparison

**Size:** L for controlled evaluation. No additional architecture development belongs in this phase.

Use the eight target rows and separate sync/async ablations from [benchmark_contract.md](benchmark_contract.md).
Run C1/C4 first. Expand to C8/C16 and safe C32/C64 only after capacity tests.
Run open-loop arrivals and frozen SLO limits. Use the 400-question quality anchor before any full-1819 extension.
Unsupported arms remain BLOCKED. Do not invent table values to complete the matrix.

**G6:** A final GO needs qualified quality, timing, and a repeatable advantage against the strongest same-resource baseline.
A throughput-only result must not become a B1-latency claim.

## Minimum Phase 1 implementation request

The next approved task should implement only these future files:

- A provenance and GPU-ownership collector.
- A same-backend TP1/TP4 bounded-fidelity harness.
- A natural-generation HTTP client with arrival and streaming timestamps.
- An offline accuracy, memory, and paired-confidence-interval analyzer.
- A small manifest that fixes calibration and held-out requests before candidate results.

These are proposed files, not existing implementations.
Do not change Ouro forward, the gate, PP, cache allocation, or the scheduler in Phase 1.
Required outputs are native qualification, numerical attribution, C1/C4 latency, memory, and a G1 decision.
Only that evidence can justify the PP2 adapter request.

## Review package

[Architecture](architecture.md), [integration map](integration_map.md), [compatibility](compatibility_matrix.md),
[benchmark contract](benchmark_contract.md), [experiment matrix](experiment_matrix.csv), [gates](acceptance_gates.md),
[risks](risks.md), [dependency lock](dependency_lock.md), [reproduction plan](reproduce_plan.md), and [evidence](evidence_index.md).
