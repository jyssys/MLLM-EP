# Looped MoE expert affinity and communication-aware EP

IN_PROGRESS: this is a preview, not a completed PoC.

## Verdict

**PENDING**

The study does not establish a new loop-aware EP architecture with at least 1.15× actionable E2E headroom.
This is not a proof that every loop-aware EP method lacks headroom.
The strongest production generic EP comparator and a validated placement wall-time model remain unavailable.
The narrow prototype gate therefore remains unmet.
The original negative results remain unchanged.

| Phase | Status | Evidence or limitation |
| --- | --- | --- |
| P0 | PASS | Pinned native checkpoints, source revisions, FA3, and TRUE EP topology. |
| P1 | CONDITIONAL | Check full anchor scores and loss margins. HellaSwag distribution unavailable. |
| P2 | PASS | 90 clean mandatory cells. Heavy observers remain separate. |
| P3 | PASS | Prompt-held-out co-activation and recurrent transition statistics. |
| P4 | BLOCKED | No validated wall-time oracle above strong production generic EP. |
| P5 | NOT_RUN | The 1.15× actionable generic-relative prototype gate is not established. |
| P6 | IN_PROGRESS | Reports retain unavailable capabilities and incomplete comparisons. |

## Models and exact execution

| Model | Learned parameters | Native topology | Experts | MoE calls/token |
|---|---:|---|---|---:|
| IFM Ouro | 8,780,889,600 | 28 physical layers repeated four times | 25/site, top-2 | 112 |
| IFM Huginn | 8,783,251,968 | 8 prelude, 12 core repeated eight times, 8 coda | 25/site, top-2 | 112 |
| Foil-1 | 553,730,048 | One prelude, one core repeated sixteen times, one coda | 8/64/8, top-2 | 18 |

All eight H100 GPUs participate in actual EP8 runs.
EP4 uses four actual ranks. Two EP4 evaluation groups are not an EP8 measurement.
Native attention, routers, top-2 score weighting, recurrent state, and BF16 weights remain unchanged.
Every expert has a real resident owner. Dispatch, owner-local expert GEMMs, and combine execute on GPUs.
The bridge also restores replicated dense hidden states after each MoE invocation.

These official execution paths recompute the full prefix for every generated token.
No retained KV cache exists in the measured adapter path.
The measured TPOT is therefore not cached single-token decode TPOT.
This distinction particularly limits conclusions about low-request cached serving.

## Accuracy evidence

| Model | EP | Benchmark | Correct/N or status | Accuracy % | Wilson 95% CI % | Paper % | Delta pp | Reference check |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ifm_ouro | 1 | MMLU_5shot_micro_acc | 8979/14042 | 63.944 | [63.146, 64.734] | 64.030 | -0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 1 | ARC_Challenge_0shot_acc_token | 599/1165 | 51.416 | [48.546, 54.277] | 51.330 | 0.086 | WITHIN_DECLARED_TOLERANCE |
| ifm_ouro | 1 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 54.820 | NOT_AVAILABLE |  |
| ifm_huginn | 1 | MMLU_5shot_micro_acc | 9379/14042 | 66.792 | [66.009, 67.567] | 66.810 | -0.018 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 1 | ARC_Challenge_0shot_acc_token | 592/1165 | 50.815 | [47.947, 53.679] | 50.470 | 0.345 | WITHIN_DECLARED_TOLERANCE |
| ifm_huginn | 1 | HellaSwag_0shot | UNAVAILABLE_OFFICIAL_DISTRIBUTION | NOT_AVAILABLE | [NOT_AVAILABLE, NOT_AVAILABLE] | 55.100 | NOT_AVAILABLE |  |
| foil1 | 1 | lambada_standard_0shot_acc | 1632/5153 | 31.671 | [30.415, 32.954] | 31.830 | -0.159 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 1 | lambada_openai_0shot_acc | 2056/5153 | 39.899 | [38.570, 41.243] | 39.980 | -0.081 | WITHIN_DECLARED_TOLERANCE |
| foil1 | 1 | xwinograd_en_0shot_acc | 1603/2325 | 68.946 | [67.035, 70.795] | 68.820 | 0.126 | WITHIN_DECLARED_TOLERANCE |

Full native-versus-EP outcomes, paired intervals, and Foil loss appear in `accuracy_reproduction.md`.
Generation output differences detected in the matched hardware matrix: **False**.
See `generation_fidelity.csv` for completed and pending cells.
No implemented loop-aware candidate has a separate accuracy score.

## Actual timing at B1 and prefix 128

| Model | EP | Prefill ms | TPOT ms | 32-token E2E ms | Output tokens/s | Static/generic E2E |
| --- | --- | --- | --- | --- | --- | --- |
| ifm_ouro | 4 | 311.692 | 317.686 | 10162.157 | 3.149 | 0.978× |
| ifm_ouro | 8 | 317.768 | 328.200 | 10497.931 | 3.048 | 1.007× |
| ifm_huginn | 4 | 367.431 | 322.281 | 10311.558 | 3.103 | 0.973× |
| ifm_huginn | 8 | 325.614 | 324.210 | 10368.953 | 3.086 | 0.982× |
| foil1 | 4 | 46.865 | 48.118 | 1540.144 | 20.777 | 0.979× |
| foil1 | 8 | 49.046 | 50.702 | 1625.627 | 19.683 | 0.970× |

These values are actual warmed 32-token greedy hardware rollouts with EOS disabled.
They are not client-to-server natural serving latency.
Each cell has five forward repetitions and three generation repetitions.
Intervals describe serial repetitions within one process, not independent server starts.
The complete 90-cell table appears in `benchmark_summary.md` and `baseline_e2e.csv`.
Additional counterbalanced process runs appear in `counterbalanced_generic.csv` when complete.

## Whole-E2E component percentages

Read **`e2e_breakdown.md`** for additive component milliseconds and percentages.
The table uses a complete observed endpoint timeline.
It does not sum rank kernel service times or extrapolate a single forward.
Dispatch and combine spans include rank waiting.
Expert spans can include submission gaps and local layout work.
The clean latency, observer latency, timing gate, and denominator remain separate.
Failed timing gates do not produce claims about clean E2E fractions.

## Recurrent affinity

| Model | Lag-1 cells | Conditional Recall@2 | Pass/domain popularity Recall@2 | Increment |
| --- | --- | --- | --- | --- |
| ifm_ouro | 5040 | 0.555 | 0.457 | 0.098 |
| ifm_huginn | 5040 | 0.550 | 0.424 | 0.126 |
| foil1 | 900 | 0.428 | 0.204 | 0.224 |

The table averages site, pass, domain, batch, and context cells uniformly.
It is a descriptive mean, not a pooled independent-prompt significance test.
Each prefill cell has sixteen held-out prompts per domain.
Calibration and held-out prompts are disjoint.
Per-cell bootstrap intervals preserve the prompt as the resampling unit.
The null preserves both top-2 slot marginals and distinct expert choices.
Finite-sample mutual information remains biased. Matched shuffle controls expose that bias.

The transition signal is real predictive evidence.
It does not make a future hidden state available.
It does not prove an exact future dispatch split matrix.
See `full_count_prediction.csv` for full-matrix tests when available.

## Why locality is not E2E gain

The original dispatcher sends one row per selected expert edge.
Two experts on the same GPU still receive two rows.
Co-placement alone therefore does not implement destination deduplication.
The fixed-placement trace model often trades lower fanout for higher rank-load imbalance.
Its transition-augmented graph can also add edges that carry no physical state in this bridge.
All MoE outputs return to a replicated dense state before subsequent computation.
There is no direct expert-to-expert hidden-state channel between recurrent passes.

The measured generic control selects an existing hidden-state copy using the exact current router.
It uses no recurrence information and moves no KV state.
Reduced remote payload does not consistently reduce its complete E2E time.
Dynamic source selection, metadata synchronization, local layout, and rank imbalance all remain possible causes.
This study does not isolate each cause enough to assign the entire result to one operation.

## Stronger generic metadata control

The original bridge already replicates exact current router outputs on every rank.
A generic control derives the full source/expert count matrix from those current outputs.
It removes the redundant count All-Gather, not expert work or authoritative routing.
Every diagnostic checks the matrix against an actual collective result before clean timing.
No future route, KV migration, or recurrent predictor enters this control.

| Model | EP | Policy | Matched static/control speedup | E2E ms | Token-exact to static |
| --- | --- | --- | --- | --- | --- |

An empty table means that these GPU controls did not complete yet.
The full result is `generic_metadata_controls.csv`.
`strongest_measured_generic.csv` selects the lowest observed mean among exact-output controls.
This selection is not a held-out performance certification or a production DeepEP comparison.
Generic metadata savings do not establish recurrent novelty.
These controls do not receive invented downstream accuracy scores.

## Headroom and novelty limits

`placement_oracles.csv` separates live controls from fixed-placement traffic and load models.
The generic and recurrent maps preserve per-rank expert counts and weight capacity.
The search is a finite swap heuristic, not an exact optimal oracle.
Initialization traffic is structural byte accounting, not a measured transfer latency.
No per-loop weight migration receives zero cost.
The study does not report a fabricated generic-to-loop wall-time speedup.

Co-activation placement, destination deduplication, and inter-layer affinity already have generic MoE precedents.
Recurrent correlation alone does not establish novelty beyond those mechanisms.
See `prior_overlap.md` for primary sources and the exact novelty boundary.

## Next research decision

Do not build a large recurrent EP engine from these results alone.
First obtain a cached native path and a stronger generic dispatcher if low-request serving remains the target.
Then profile complete E2E intervals and identify a legal state channel or removable metadata dependency.
Any successor must compare the same memory budget, exact expert work, and GPU count.
It must preserve benchmark quality and demonstrate incremental wall-time gain above the generic control.
