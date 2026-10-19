# Tile dataflow prior audit

Status: primary-source and selected-code audit complete. Ouro integration, numerical parity, and GPU performance are `NOT_RUN`.

## Outcome

Tile execution can remove unnecessary whole-operator barriers without using stale inputs. Legal examples include complete-neuron SwiGLU tiles and completed-feature norm statistics. These examples can expose within-token work at concurrency one. They are not merely independent-request batching.

The inspected mechanisms nevertheless overlap generic tile fusion, fine-grained dependency scheduling, and persistent task execution. No recurrence-specific execution axis survives this source screen. This verdict does not reject useful kernel engineering or a future measured advantage.

R4 repeats the optimized body. Repetition alone does not create the next pass's input sooner. A stronger claim needs a benefit that remains after generic tile execution and aliased logical unrolling receive the same ready work.

## Scope and duplication check

The target remains Base Ouro-1.4B/2.6B, BF16, fixed R4 or the original adaptive gate, without training. The adaptive contract retains q=0.5, minimum R2, maximum R4, full-R4 prefill, and the selected historical KV policy.

Before searching, this worker read the current persistent-execution, ownership, exact-frontier, original-gate, mixed-depth-frontier, and layout reports. Those reports already cover persistent RNNs, MPK, NanoFlow, generic collectives, and mixed-depth batching. This audit does not count those mechanisms as new discoveries. It adds four compiler/dataflow sources and inspects additional MPK runtime lines.

The [source and search observations](../artifacts/tile_dataflow_sources.jsonl) preserve versions, selected sections, code revisions, access times, and failed lookups.

## Primary-source comparison

| ID | Source and inspected version | Mechanism and transfer limit |
|---|---|---|
| TD01 | [Rammer, OSDI 2020](https://www.usenix.org/system/files/osdi20-ma.pdf), Sections 3.1–3.3 and 4.1 | Persistent block workers execute task programs with explicit waits. The default policy waits for all predecessor-operator tasks. |
| TD02 | [Welder, OSDI 2023](https://www.usenix.org/system/files/osdi23-shi.pdf), selected Section 3 passages and artifact appendix | Connected operator tiles reuse intermediates. Input-region propagation and resource limits constrain fusion. |
| TD03 | [Mirage, 2405.05751v3](https://arxiv.org/html/2405.05751v3), 2025-06-06, Sections 3, 5, and 6 | Multi-level graphs combine algebra, scheduling, layouts, and memory planning. Its verifier does not prove BF16 bitwise equality. |
| TD04 | [Kitsune, 2502.18403v1](https://arxiv.org/html/2502.18403v1), 2025-02-25, Sections 4–6.1 | L2 queues connect spatial stages. Modified grid scheduling and simulated end-to-end results limit direct H100 transfer. |
| TD05 | [MPK runtime, ca58a02](https://github.com/mirage-project/mirage/blob/ca58a02d110a278bffaed503be5e66494db2597d/include/mirage/persistent_kernel/persistent_kernel.cuh) | Existing generic control. This round inspected iteration IDs and dependency-event publication, not a new full paper audit. |

### Rammer: distinguish interface capability from default scheduling

The paper exposes `Append` and `Wait` over individual tasks. Algorithm 1 instead waits for `Predecessor(op).rTasks` before each consumer task. Thus its default policy does not establish automatic producer-tile streaming for every adjacent operator.

The pinned [BlockParallelDevice](https://github.com/microsoft/nnfusion/blob/00c13173a155556e50510575f6682a6a654cd585/src/nnfusion/engine/pass/graph/blockfusion/block_parallel_device.cpp#L136) inserts group waits and execution instructions. The [code generator](https://github.com/microsoft/nnfusion/blob/00c13173a155556e50510575f6682a6a654cd585/src/nnfusion/engine/pass/graph/blockfusion/blockfusion_codegen.cpp#L373) separates a deduplicated kernel body from each call's input/output arguments. Generic code reuse therefore does not require activation reuse. The inspected fused emitter registers FP32, not an established BF16 Ouro path. No build or correctness test occurred.

### Welder: tile shape, reduction, and lifetime are explicit costs

`SetConnect` and `Propagate` connect adjacent operators through a common tile. Section 3.1 retains necessary input regions and reduction-axis accumulation. This is direct prior for consuming a ready intermediate tile before materializing the full intermediate tensor.

The official artifact branch is `microsoft/nnfusion:osdi2023welder`, revision `7d1e71e`. Its [IRNode](https://github.com/microsoft/nnfusion/blob/7d1e71ea2b5acedd7e87be572523a8cbf1fc4e82/python/welder/graph.py#L185) propagates input/output and reduction regions. Its [policy](https://github.com/microsoft/nnfusion/blob/7d1e71ea2b5acedd7e87be572523a8cbf1fc4e82/python/welder/policy/default.py#L283) tracks intermediate lifetimes. `compute_tile_dict` rejects excessive shared-memory or register requirements. The published artifact targets V100. This audit does not establish a ready BF16 H100 or Ouro adapter.

### Mirage: relevant fusion, different numerical contract

Section 3 computes RMSNorm statistics and an unnormalized matrix product in parallel. It then divides the product by the norm. This is a real-arithmetic reassociation, not elimination of the norm's data dependency. Equation 1 also does not specify Ouro's complete epsilon and intermediate-cast contract.

The [reference graph](https://github.com/mirage-project/mirage/blob/ca58a02d110a278bffaed503be5e66494db2597d/demo/reference_mugraphs/rms_norm.py#L11) separately accumulates matmul and RMS, then divides. This inspected example uses FP16. Section 5 verifies Lax-program equivalence through finite fields. Its numerical-stability paragraph separately uses floating-point tests. Neither step proves equality with Ouro's BF16 rounded graph. The [earlier layout report](layout_and_structured_residual_redteam.md) already demonstrates why removing intermediate casts needs separate qualification.

### Kitsune: software queues do not imply available hardware scheduling

Sections 4–5 use sequence-numbered ring queues, CTA barriers, and stage resource allocation. Consumers wait for produced tiles. Reductions split into fan-in and final stages. Section 6.1 reports A100-class simulation with a modified scheduler. Queue microbenchmarks ran on silicon, but they are not end-to-end H100 results.

A newer [TACO article](https://doi.org/10.1145/3777466) exists. Publisher metadata dates online publication to 2025-12-15. Its primary page and PDF returned HTTP 403. Therefore, this report's mechanism claims refer only to the inspected v1 body. The latest journal body is `UNAVAILABLE_TO_THIS_AUDIT`. Official code was `NOT_LOCATED` in the bounded search. An educational third-party implementation does not close that gap.

### MPK: concrete generic completion and version controls

The existing campaign already identifies MPK as a strong persistent-runtime control. The current official revision remains `ca58a02`, dated 2026-10-17.

This round read `compute_task_id`, lines 142–152, and the worker event path, lines 1087–1190. Task IDs include an iteration and graph position. Workers wait on dependency counters before execution. The local path uses an acquire load and a release atomic increment around task completion. An NVSHMEM event path is also present. These are relevant generic mechanisms, not a proof that an Ouro graph or its state publication is already supported.

## Legal tile frontier for Ouro

The following table is an `ARCHITECTURAL_INFERENCE` from the inspected operator sequence. It does not assert achieved concurrency.

| Region | Legal early consumer | Required completion condition | Boundary that remains |
|---|---|---|---|
| Gate/up projections | SiLU and product for neuron tile J | Both complete projections for J use the actual normalized input | An incomplete dot product cannot enter SiLU |
| SwiGLU output | Down-projection work for completed J | The rounded product tile exists and its version remains live | Every required down contribution precedes final output norm |
| Attention heads | Per-head attention and output-projection work | Matching complete Q/K/V head data and causal history exist | The complete head read still needs all visible attention contributions |
| Completed output features | Partial RMSNorm statistics | Each included feature already has its final projection/reduction value | The normalization scale needs every required feature statistic |
| Loop-normalized output | Gate work and next input-norm statistics | Actual normalized feature tiles exist | Next-pass projection needs the required normalized input and execution guard |

For SwiGLU, write complete intermediate tiles as:

```text
g_J = W_gate[J,:] x
p_J = W_up[J,:] x
z_J = SiLU(g_J) elementwise_multiply p_J
d_J = W_down[:,J] z_J
d   = sum_J d_J.
```

`d_J` can start after its own `z_J` exists while another neuron tile remains unfinished. Final `d` still contains every required contribution. Ouro's output norm consumes completed `d`, not arbitrary partials. Producer/consumer fusion is legal at this granularity. A different summation order can still change floating-point results.

The local native source confirms four block norms and a recurrent boundary norm. See `OuroDecoderLayer.forward:260–281` and `OuroModel.forward:354–375` in the pinned installed model. Its residual has two stored parts. A tile scheduler must preserve both parts and their lifetime, rather than treat one stored tensor as the complete residual.

Full-row reductions constrain the frontier, but they do not require materializing every full tensor in HBM. A fused implementation may stream correctly rounded tiles after the necessary statistic exists. That is a different claim from normalizing an unfinished feature subset or moving a division through a BF16 cast.

At the outer boundary, the next pass still depends on the current pass's actual normalized result. Minimum R2 permits the already-audited R1 gate overlap. Eligible-depth continuation still needs a committed decision or the existing provisional-state contract. Tile scheduling does not remove original-gate or KV ownership rules.

## Persistent shared weights: what repetition changes

Weight identity and activation identity are separate. A generic unrolled graph can assign the same immutable weight pointer to repeated operator instances. It can retain separate request, token, layer, depth, and KV-version identifiers.

Holding a selected weight tile across useful ready rows can amortize loads. A generic tile executor can use the same grouping. At concurrency one, the next use of a layer in the next pass still follows the intervening layers. Persistence does not make that next input ready.

The existing [capacity audit](persistent_execution_redteam.md) rejects complete ordinary BF16 core residence under its stated storage bound. It explicitly leaves streamed persistence and partial retention open. This report does not repeat that arithmetic or strengthen it into a ban on all weight reuse.

For a named retained subset, compare the same cache policy, weight pointers, ready rows, and lifetime limits in the generic baseline. Separate saved HBM traffic from shorter dispatch and from improved occupancy. No new traffic counter or reuse-distance measurement occurs here.

## Decision and cheapest useful test

The candidate classification is `GENERIC_TILE_DATAFLOW_WITH_UNMEASURED_HEADROOM`. A new-axis claim is `NOT_ESTABLISHED`.

A useful implementation test could target source-order SwiGLU-to-down streaming or completed-feature norm statistics. It must compare against competent fused kernels and a generic tile executor. It must preserve BF16 casts, actual gate hazards, residual ownership, and depth-indexed KV identities.

The comparison must charge:

1. Producer/consumer queue operations and publication fences.
2. Tile scratch, retained accumulators, and live residual versions.
3. Resource partitioning, register pressure, occupancy, and launch changes.
4. Extra reads, gathers, reductions, and any reordered arithmetic.
5. Gate routing, discarded provisional work, and state commit.

First identify a tile edge with an unnecessary baseline wait. Then determine whether the saved exposed interval exceeds those costs. Do not infer latency from a larger mathematical frontier alone.

Use an R1 control with the same operator shapes, plus aliased logical R4 unrolling. A gain shared by both is generic kernel/runtime engineering. A tied-weight-specific gain needs a separate retained-byte or setup-reuse mechanism and a matched control. None of the inspected sources establishes that gain for both Base sizes or both depth policies.

## Evidence and time

This worker performed read-only source retrieval, selected local source inspection, and report validation. It executed no repository, model, or GPU kernel. It changed only this report and its source observations.

The segment ran from 2026-10-19T11:50:24Z to the accounting cutoff at 2026-10-19T11:57:35Z. Its observed duration was 431 seconds, or 7 minutes 11 seconds. Worker time overlaps coordinator time and does not add campaign elapsed hours. Final stamping and validation after this cutoff are outside that duration. The report passed prose validation with zero findings.
