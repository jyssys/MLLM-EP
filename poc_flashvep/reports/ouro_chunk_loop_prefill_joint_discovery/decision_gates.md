# Frozen experiment and decision gates

## Input contract

Authoritative specification SHA256:
`998fafdfbdac57ea288c7f7480c1258710c6bea890befc3d6aab6b5854be8a1e`.
The input is `poc_flashvep/reports/ouro_chunk_loop_prefill_joint_poc_spec.md` on the server.
The manifest froze inputs, trial IDs, chunk choices, groups, and thresholds before the candidate grid.
See `workload_manifest.json` for its content hash.

## Primary settings

- Base Ouro-1.4B, BF16, Static R4 for prefill and decode.
- Physical GPU4. GPUs5–7 retain unrelated permitted-task burners.
- Prompt bins near2K/8K, concurrency4/8, token budget1024.
- Chunk menu128/256/512/1024. Groups1/2/4.
- Natural greedy generation, native EOS allowed, output cap128 for the interference workload.
- Mathematical background prompts and long prompts constructed from public natural problem text.
- Calibration trials0/1. Frozen future held-out trial IDs2–15.

The long prompts are constructed natural-text workloads, not unmodified conversational traces.
The cap128 is a serving stress setting, not a sufficient reasoning accuracy budget.

## SLO and measurement

The initial SLO is TTFT <=5s and per-request maximum observed ITL <=100ms.
The SLO is fixed. Do not relax it after candidate results.
Report all-zero qualified-completion cells instead of inventing a useful threshold.
For a drained burst cohort, qualified completions/cohort-duration is not steady-state open-loop goodput.
Open-loop goodput remains a distinct conditional experiment.

The primary interference workload primes background decode before scheduling a long prompt.
It schedules the long arrival20ms after the common decode-readiness origin.
Actual host admission can occur later at an engine boundary.
Report this causal arrival rule. Do not call it an unmodified external arrival trace.

The full request clock includes background prefill, graph activity, queueing, generation, and output delivery.
The separate interference-span clock excludes priming.
Never use the interference span as full cohort makespan without that qualifier.

## Phase gates

| Gate | Requirement |
| --- | --- |
| G0 | Source audit, working pretrained Atomic/Split/Mixed, bounded exact reference, and stock tests |
| G1 | Minimal G2 patch, full-R4 counts, valid per-depth KV, hidden/logit/token checks, and lifecycle tests |
| G2 | Repeatable useful static frontier beyond tuned Atomic/Split/Mixed, with uncertainty and memory accounting |
| G3 | Adaptive Joint beats adaptive C-only/G-only and strong stock controls, using only causal state |
| G4 | Actual two/four-GPU execution with matched replica baselines, if earlier gates warrant expansion |

The practical G2 threshold is >=5% in the registered ITL-p95/TTFT trade-off or >=10% goodput.
An isolated lower mean does not pass.
If intervals include no benefit, report CONDITIONAL rather than implementing an adaptive scheduler automatically.
If stock tuning dominates, stop before Phase3 and Phase4.

Use request/cohort/run bootstrap, never independent token or kernel resampling.
Shortlist cells should use at least ten paired cohorts and three clean process starts when practical.
Low-sample P99 remains exploratory.
The maximum ITL and SLO violation count are required because rare large stalls can escape p95 or p99.

## Quality gates

Same-backend fixed-C grouping should preserve exact outputs where numerical execution order permits.
Record hidden/logit/KV errors separately from output agreement.
The pre-registered diagnostic relative-L2 limits are0.03 for hidden, logits, and KV.
Exact equality is the stronger preferred result.
A full accuracy non-regression claim requires complete natural answers and adequate paired benchmark samples.
Do not infer that claim from truncated serving answers or short parity probes.

No policy may skip prompt loops, use future generated text, or change model precision.
No method may hide a graph miss or graph preparation that occurs during request service.
No multi-GPU result may relabel independent replicas as TP or PP.
