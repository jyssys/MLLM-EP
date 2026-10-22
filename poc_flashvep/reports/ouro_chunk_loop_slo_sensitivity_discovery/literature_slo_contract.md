# Literature and SLO contract

## Directly checked definitions

| Source | Relevant evaluation rule | Use in this study |
| --- | --- | --- |
| [Layered Prefill, MLSys 2026, Table 5 and Section 5.2](https://proceedings.mlsys.org/paper_files/paper/2026/file/c0f460c6d63599ea870ba9db63dc96a9-Paper-Conference.pdf) | A request must meet TTFT and every subsequent TBT limit. Capacity curves use a 90% attainment reference | Separate request-max compliance from pooled-token quantiles. Calibrate Ouro limits rather than copying another model |
| [Sarathi-Serve, OSDI 2024, Section 5](https://arxiv.org/html/2403.02310v3) | Median TTFT, pooled p99 TBT, Poisson arrivals, and sustainable load. Strict/relaxed targets use 5×/25× no-interference decode time | Preserve its population-tail objective as a separate metric. Do not relabel it as request-max compliance |
| [SLOWeave, September 2026](https://arxiv.org/html/2609.07883v1) | Select a token chunk using observed decode deadlines and a monotone iteration-cost model | Strong adaptive C-only methodological control. It is not a measured RLT baseline here |
| [FlowPrefill, February 2026](https://arxiv.org/abs/2602.16603) | Operator boundaries and event-driven scheduling reduce blocking without constant fine-grained scheduling | Existing granularity/overhead idea. Do not claim scheduling boundaries alone are novel |
| [P-PAS, revised October 2026](https://arxiv.org/abs/2608.15171v2) | Load-dependent token-budget adaptation using prefill/decode pressure | Another C-only mechanism. It does not establish joint recurrent grouping |
| [DuetServe, ICML 2026](https://proceedings.mlr.press/v306/gao26m.html) | Adaptive SM partitioning isolates prefill and decode within a GPU | Distinct spatial multiplexing. This PoC does not reproduce it |
| [CDB, September 2026 revision](https://arxiv.org/abs/2608.09444v2) | Continuous recurrent-depth decode scheduling | Prior recurrent execution. It is not an online C/G SLO result |
| [Pinned RLT policies](https://github.com/InferMatrixLab/vllm-rlt/blob/4a350344027960a48384626aa646f1765fda6d4f/vllm_rlt/core/scheduling_policy.py) | Atomic G4, Split G1, and compatible Mixed recurrent batches already exist | All remain strong baselines. The old G2 adapter is reused unchanged |

Layered Prefill uses 5/10 s TTFT limits and 100/125 ms TBT limits for its particular model/workload pairs.
Those are not universal latency requirements.
Its MoE weight-reloading benefit does not transfer automatically to dense Ouro.
Sarathi's reference context/batch is 4K/B32, unlike the initial Ouro calibration.
We report the actual Ouro context, batch width, and measured denominator.

## Metric separation

Request TTFT starts at the scheduled open-loop arrival, not delayed server admission.
TBT is an actual interval between delivered consecutive tokens.
A request-max rule requires every such interval to meet its limit.
A pooled p99 rule can tolerate rare violating intervals and can therefore disagree with request compliance.
The analysis reports both contracts without treating one as the other.

All offered requests count in request attainment.
Errors, missing completions, and timeouts fail joint attainment.
Goodput divides successful offered-window requests by the declared arrival-window duration.
Drained-cohort throughput uses a different, explicitly labeled denominator.
An increasing queue does not establish sustainable capacity.

## Recent-work search boundary

Targeted searches also checked Ouro prefill/loop/SLO scheduling and recurrent group scheduling.
They do not prove the absence of another C/G method.
No priority claim follows from an unlocated paper.
Live Adaptive Joint remains conditional on evidence beyond tuned fixed C-only, G-only, and Mixed.
