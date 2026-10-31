# Dynamic A: Admission Gate and Headroom

Status: FINAL_NOT_RUN_GATE_NOT_MET.
Phase B and its automatic analysis completed.
No new Dynamic A GPU campaign started.
The final review did not find a qualified tuned P-compute/D-spare condition.

## Required gate

Phase C requires a measured P-compute-limited regime.
The same regime must retain D compute capacity, valid KV credit, and Decode SLO margin.
An isolated service-rate ratio is insufficient.
An N24 Prefill queue alone is also insufficient.
The current study keeps the original gate.
It does not relax the gate to obtain a positive result.

`bottleneck_screening.csv` is a screening analysis.
Its `P_candidate` flag does not authorize a GPU method campaign.
Its `C_GPU_GATE_AUTHORIZED` flag therefore remains false until an evidence review authorizes a specific condition.

## Current model evidence

| Model | Internal baseline qualification | Relevant observed limit | New online partition GPU result |
|---|---|---|---|
| Ouro R4 | 32-request, partial-tail, and enlarged-setting gates pass | Generic chunk tuning removes tested queues. The enlarged G3 engine loses D SLO and KV margin. | NOT_RUN_GATE_NOT_MET |
| Huginn R32 | 32-request, partial-tail, and enlarged-setting gates pass | All 22 classified settings satisfy the declared D-KV-admission criterion. The rate sweep is complete. | NOT_RUN_GATE_NOT_MET |

These qualifications do not establish original-HF equivalence or downstream accuracy.
No current correctness failure justifies suppressing either model's characterization.
The performance gate is a separate issue.

## What has not been implemented or measured

- No new live Loop-Dynamic policy uses the current study's measured service data.
- No new Token-Dynamic policy uses those data.
- No new fixed-cut comparison has established an offloading benefit on this grid.
- No current full-cut Snapshot control has established a transport-only gain.
- No contention-validated event simulator predicts Dynamic A goodput.
- No oracle speedup is available from the current capacity ratio alone.

The previous study's online decisions remain historical evidence.
They are not new decisions from this study.
Native PD uses a full cut by construction.
Its zero continuation count must not be presented as a failed new Dynamic policy.

## Why the simple capacity oracle is not valid

Prepared simultaneous Decode batches can be much more efficient than staggered requests.
The controlled release diagnostic demonstrates this difference without timed Prefill or incoming transfer work.
Actual KV reservation also couples P admission to D resources.
The service rates therefore cannot be combined as independent constant-rate servers without validation.

A valid oracle must model at least these factors:

1. P service and actual P queues.
2. D active rows, phase alignment, and token completion.
3. Reserved and valid KV state.
4. Handoff, admission, and synchronization.
5. Runtime control work and its interaction with GPU submission.

The study does not sum independent per-request minima as a Serving oracle.
It does not convert FLOP fractions into measured goodput.
It also does not derive a speedup from summed host E2E percentages.

## Conditions for a later implementation

The existing exact Loop and Token connectors remain available.
They preserve valid KV, hidden state, and Huginn injection state.
They must receive another qualification if a new geometry or legal frontier changes their execution.

A later policy must use isolated service costs rather than queue-contaminated TTFT.
It must account for actual D readiness and batch packing.
It must recheck admission at the handoff frontier.
If admission fails, P must safely finish the remaining work.
The policy must log accepted continuations, rejected continuations, and source fallbacks.
It must not assume future arrivals or true future output lengths.

All method comparisons require the same model, numerical corrections, GPU count, and workload.
They also require matched graph and memory policies.
Native chunk streaming and full-cut Snapshot are distinct protocols.
A matched full-cut control must separate protocol gains from partition gains.

The strongest comparison includes DP2 and DP2 with separate-phase chunking.
That chunked baseline is not a production Sarathi-Serve implementation.
Shared-GEMM Mixed batching remains outside this matched axis comparison.

## Structural Loop-versus-Token considerations

The following argument is analytical structure, not measured headroom.
Let `f` be a prefix-token fraction or a recurrent-depth fraction.
Ignore boundary work and assume a long causal sequence.
Linear and FFN work assigned to P scales approximately as `f` in both cases.
Attention pair work assigned to P scales approximately as `f^2` for Token Split.
It scales approximately as `f` for Loop Split.

At a half split, the corresponding P/D attention pair fractions are approximately 25/75 and 50/50.
These are not GPU-time percentages.
Token Split can select another prefix fraction.
The result therefore shows different work mixtures, not universal Loop dominance.

Token Split can retain a large chunk budget.
It does not inherently require small GEMMs.
The current exact token connector can clip the final task at a prefix frontier.
An unnecessarily coarse prefix action space would weaken that baseline.
It would not establish a loop-specific advantage.

Loop continuation and Token suffix continuation also have different immediate data dependencies.
Their total transfer cost and critical path still require measurement.
The study does not claim a measured activation-first or deferred-KV optimization.

## Novelty boundary

A generic logical-layer action space includes recurrent boundaries.
At the same logical boundary, the mechanisms can be equivalent.
Both ordinary PD workers can already hold a complete model.
Shared weights therefore do not automatically provide another model-residency gain over that baseline.

Generic prefill borrowing is also established prior work, including [FluidPD](https://arxiv.org/abs/2610.06917).
Neither an adaptive cut name nor Native Mixed batching proves a new method.
This study excludes Layer Split implementation.
It cannot establish superiority over a complete generic Layer oracle.

## Primary and secondary conditions

The primary output grid remains 128–256 tokens, with separate 448-token controls.
The specification permits a 3,500/32 diagnostic if the primary screen finds no eligible crossover.
Such a diagnostic cannot replace the primary grid.
Any positive short-output result must remain separate from primary method claims.

## Final decision

The complete available Phase B evidence does not authorize Phase C.
The two models have independent `NO_P_HEADROOM_ON_TESTED_GRID` decisions.
The new online controllers remain `NOT_IMPLEMENTED_GATE_NOT_MET`.
No valid new Dynamic A speedup or oracle speedup exists in this campaign.

The final [method-status CSV](looped_pd_bottleneck_crossover/phase_c_method_status.csv) records every omitted comparison.
The [final review](looped_pd_bottleneck_crossover/state/final_phase_decisions.json) supersedes the preserved provisional gate for interpretation.
It does not overwrite earlier source or raw evidence.

This decision does not prove that no Prefill-heavy regime exists.
The measured Full-Prefill PD baseline still divides Prompt tokens into chunks.
Whole-prompt atomic PD, exact 4K/100, and Ouro 8K conditions remain unmeasured.
The study therefore cannot establish an all-context or architecture-level impossibility.
Its architectural conclusion is `INCONCLUSIVE`.

The next useful work is baseline validation, not immediate controller complexity.
First compare whole-prompt PD with tuned chunked PD at matched transport and memory settings.
Then test legal longer-context conditions and validate real D reserve.
Only a positive measured gate should trigger another static Loop/Token offloading comparison.

The absence of a new policy result is not a measured performance failure of that policy.
It must not be reported as `NO_GO_DYNAMIC_A` or universal Loop-Split inferiority.
