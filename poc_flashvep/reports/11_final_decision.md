# Final decision — refinement-aware and EP-aware expert allocation

Verdict: `NO_GO` for the tested allocation design and current native EP4 bridge.
The follow-up completes the feasible discovery checks and applies the specification's staged stop conditions.
It does not establish that all refinement-aware allocation methods are impossible.
Model and runtime identities appear in `00_runtime_audit.md`.

| Phase | Current status | Evidence |
|---|---|---|
| 0 Runtime audit | PASS for Groups A and B | Model revision, ownership, nonzero remote assignments |
| 1 Quality reproduction | CONDITIONAL_PILOT_COMPLETE | GSM8K 128 and HumanEval 32. Reference protocol remains non-identical |
| 2 Refinement sensitivity | NO_GO_ALGORITHM_FOR_TESTED_PROXY | 96 free-running interventions. P3 does not improve P2 correctness |
| 3 EP cost characterization | COMPONENT_MEASURED_GATE_NOT_PASSED | 36 clean physical operator conditions, 30 repeats each. No faster candidate mean |
| 4 Sparse generation runtime | NOT_RUN_AFTER_GATES | Sparse operator executes, but no full-generation engine expansion |
| 5 Candidate benchmark and serving | NOT_RUN_AFTER_GATES | No P5 test-quality or request-speedup claim |
| 6 Ablation and final decision | DISCOVERY_ABLATIONS_COMPLETE | Budget, routing/refinement, NFE, no-op, hardware, and operator controls. Wider robustness grid not run |

Phase 3 does not supply a validated full-request predictor.
Its operator evidence does not justify sparse-generation engineering.
The campaign does not claim a calibrated universal E2E upper bound.

## What is established

The exact non-preview Mini checkpoint loads on the available hardware.
Group A performs physical expert-sharded EP4 execution.
All 32 native HumanEval pilot requests preserve output tokens and NFE through the identity hook.
The original historical quality counts match their saved sample records.
The HF and native temperature-zero paths implement different token selection.

## What is not established

Fresh full benchmark accuracy is not complete.
Group B correctness, isolated P0 repeats, and physical operator comparisons completed.
No deployed sparse candidate establishes population quality preservation.
Physical assignment reduction is demonstrated only in the standalone operator probe.
No candidate isolated E2E speedup is measured.
No incremental EP-aware gain over refinement-only allocation is demonstrated.

## Why the current design stops

The causal confidence signal predicts next commitment well, but does not establish safe expert omission.
P3 matches router-only correctness at both tested budgets.
Its extra diagnostic softmax is not free.
EP-aware structural balancing reduces owner CV, but sacrifices some message elimination.
The physical sparse operator removes real assignments, yet all 24 candidate mean ratios are below 1.000×.
The requested incremental EP benefit is therefore not established.
Full-generation P4/P5 engineering would exceed the evidence available at the stop gates.

The baseline pilot, retained-all traces, sensitivity screen, and physical cost measurements are complete.
Their state files reside in `../results/refinement_aware_ep_expert_allocation/state/`.
No full-generation sparse allocation policy is validated.
The reproduction entry point is `../refinement_ep_allocation/README.md`.
The controllers restore owned burners after their queues finish.
The earlier automatic full-evaluation and isolated-baseline queues are canceled.
Their design remains available but does not execute automatically.
See `refinement_ep_pilot_scope_amendment.md` for the current scope.
Any later sparse-policy work requires new evidence and a separate gate review.
The previous Looped MoE research remains stopped.

## Baseline accuracy and official comparison

| Dataset | Official reference | HF pilot | Physical EP4 pilot | EP4 Wilson 95% CI | EP4 minus official |
|---|---:|---:|---:|---|---:|
| GSM8K | 94.24% | 118/128 = 92.1875% | 119/128 = 92.9688% | [87.1765%, 96.2570%] | −1.2713 pp |
| HumanEval | 86.59% | 30/32 = 93.7500% | 32/32 = 100.0000% | [89.2821%, 100.0000%] | +13.4100 pp |

Model: `inclusionAI/LLaDA2.0-mini`, revision `dad945cac317da394b390f82c7b40691d8a881ed`, BF16.
HF preserves its official implementation's multinomial call at temperature zero.
The native decoder uses greedy selection and prefix KV caching.
The full published evaluation protocol is not reproduced identically.
The first-32 HumanEval subset does not establish superiority over the official score.
The original full benchmark remains deferred under the user's sample amendment.
No output cap or decoding setting changed to accelerate this follow-up.

The GSM8K EP4-minus-HF difference is +0.78125 pp, with paired 95% interval [−2.34375, +4.68750] pp.
The HumanEval difference is +6.25 pp, with interval [0, +15.625] pp.
These backend comparisons include different token-selection protocols.
They do not isolate the effect of EP sharding.

## Policy evidence table

| Policy | Measured quality evidence | Physical request acceleration | Interpretation |
|---|---|---|---|
| P0 full | GSM8K 119/128, HumanEval 32/32 | Baseline only | Functioning physical EP4 |
| P0-Hooked | HumanEval 32/32, token/NFE parity 32/32 | NOT_MEASURED | Identity control |
| P1 uniform | 15/16 at both validation budgets | NOT_MEASURED | Dense semantic diagnostic, not sparse generation |
| P2 router-only | 14/16 low budget, 15/16 high budget | NOT_MEASURED | Dense semantic diagnostic. Physical operator tested separately |
| P3 refinement proxy | 14/16 low budget, 15/16 high budget | NOT_MEASURED | No correctness advantage over P2 in this screen |
| P4 EP-only | NOT_MEASURED | NOT_MEASURED | Structural replay only |
| P5 joint | NOT_MEASURED | NOT_MEASURED | Structural replay only. Deployment gate does not pass |
| P6 hindsight oracle | NOT_RUN | NOT_APPLICABLE | No fabricated marginal-quality labels |

At the lower budget, actual semantic omission is 4.6875%.
At the higher budget, live omission is approximately 18.84%, with changed trajectories.
All diagnostic expert kernels still execute.
These semantic omission percentages are not executed-work reductions.
The 16-question screen cannot establish the 0.5 pp quality target.

## Physical findings

TRUE EP4 uses 64 distinct experts per rank across 19 MoE layers.
Dense TP4 and source-row partitioning support the physical EP4 bridge.
Groups A and B are independent replicas, not EP8.
Actual remote assignments and received buffers prove inter-rank execution.
Wire-byte hardware counters remain unmeasured.

The standalone sparse probe uses original BF16 expert weights and actual native activations.
It removes assignments before dispatch, rather than masking only a completed output.
It includes the cost of sparse-ID and weight materialization.
Its predictor mask is precomputed, so online selection costs are optimistic.
Across 24 candidate conditions, the clean operator speed ratios range from 0.9261× to 0.9980×.
Five 95% repeat intervals include 1.000×.
No point estimate demonstrates acceleration.
This is an operator negative result, not a measured negative request speedup.

P0 isolated mean request time is 5.9247 s on A and 6.4577 s on B.
Each group contributes 30 repetitions of one 1,477-input/198-output-token request.
All repetitions retain 154 forwards and identical output tokens.
Group B has large transient outliers, with P95 9.2034 s.
The experiment does not establish a hardware cause or population serving capacity.

## EP-specific contribution test

At the high replay budget, P3 reduces remote token-rank pairs by 4.4650%.
P5 reduces them by only 3.2002%.
P5 reduces owner CV from 0.39690 to 0.26617.
Its maximum-rank padded-row reduction is 15.0720%, versus P3's 12.7180%.
That extra 2.3540 percentage points concerns a structural count, not E2E latency.
Global cost metadata, selector overhead, and changed NFE remain additional costs.
No measured P5-over-P3 request improvement exists at a matched executed-edge budget.

The PoC establishes neither incremental algorithmic benefit nor physical EP-specific E2E benefit for the tested joint design.
It does not establish a new ML contribution beyond REFLEX/Epoch/position-wise computation methods.
The tested proxy and native bridge do not justify further engine implementation now.

## Important negative results and missing evidence

- High-budget uniform omission preserves aggregate 15/16 correctness, but raises mean per-question NFE by 56.71%.
- A wrong uniform request reaches the generation cap. Raw outputs and failure IDs remain preserved.
- Commitment AUROC 0.92619 does not predict marginal expert importance.
- Removing edges does not remove proportionate destination messages or padded kernel work.
- The current B1 bridge has substantial launch, metadata, and synchronization costs. Their exact causal split remains unresolved.
- A faithful REFLEX implementation and Epoch complementarity remain NOT_RUN.
- Full GSM8K/HumanEval candidate quality, batch/block expansion, and sustainable serving capacity remain NOT_RUN after the gates.
- A validated full-request cost model and physical fabric-byte counters remain NOT_MEASURED.

The next useful test is a lower-overhead physical sparse primitive, not a larger allocation engine.
It must first beat the strongest matched native operator at unchanged semantics and geometry.
Only then should a new causal signal justify another quality screen and request-level evaluation.

## Reproduction and publication

Reports `00_runtime_audit.md` through `10_prior_work_comparison.md` contain the detailed evidence.
The code entry point is `../refinement_ep_allocation/README.md`.
Results reside in `../results/refinement_aware_ep_expert_allocation/`.
Raw archive indexes retain hashes and restoration instructions.
The new `sensitivity_and_physical_cost` archive includes failed and superseded attempts.
It does not replace the preserved original files.
Editable SVG figures reside in that result directory's `figures/` folder.
Commands, model/runtime hashes, seeds, GPU maps, and failure logs remain preserved.
Initial helper-import snapshots are incomplete.
Final helper hashes do not prove the exact helper source in every earlier in-memory worker.
Driver snapshots, behavior fields, source revisions, and regression checks remain available.
Branch: `flashvep/refinement-aware-ep-expert-allocation-poc`.
The completion response provides the verified publication commit and remote SHA.
