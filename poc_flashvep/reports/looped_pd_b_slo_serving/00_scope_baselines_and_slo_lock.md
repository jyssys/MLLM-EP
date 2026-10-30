# Scope, baselines, and prospective SLO lock

Status: experiment in progress. No serving-capacity or lossless GO claim is accepted.

The parent evidence is commit `b82eabd20a4ec168a5ad9a119c8401f97a565e78`.
This follow-up uses branch `flashvep/looped-pd-b-slo-serving-poc`.
The prior A/B/C source and raw evidence remain unchanged.
Only dense Ouro Base 1.4B R4 and Huginn-0125 R32 are active.
All tokens execute the prescribed fixed recurrence in BF16.
A, C, dynamic exit, MoE, approximation, quantization, and training are excluded.

## Primary endpoint

The primary endpoint is the maximum tested sustainable open-loop arrival rate with at least 90% joint TTFT+TPOT attainment.
An incomplete rate bracket or insufficient stationarity evidence receives `BOUNDED_CAPACITY_ONLY`.
A favorable fixed-concurrency E2E point is not the primary endpoint.
Method and system comparisons remain separate.

| Method | Placement | Scheduling and transfer |
|---|---|---|
| DP | N colocated replicas | Native default fixed-depth scheduling, token chunk256 |
| DP_Mixed | N colocated replicas | Native mixed recurrent prefill/decode scheduling, chunk256 |
| PD | N/2 P and N/2 D | Completed-chunk transfer can overlap later chunks |
| PD_G1 | Same P:D | Wavefront G1, chunk256, existing completed-chunk NIXL overlap |
| B | Same P:D | PD_G1 plus immutable recurrent-frontier transfer |
| GenericLayer | Same proposed P:D | Structural novelty audit. No separately measured runtime yet |

Each GPU has the same native cache and active-request limits within its model.
The default active capacity is eight requests per worker.
The native cache uses 4096 blocks for Ouro and 12288 for Huginn, with block size16 and depth-indexed last-exited semantics.
These capacities are not interchangeable model settings.
The attention backend is FlashAttention4. Ouro uses its qualified asynchronous path. Huginn uses its qualified synchronous boundary adapter.
CUDA Graphs remain enabled. Warmup and unseen serving-time preparation remain distinct.

The P:D defaults are 1:1, 2:2, and 4:4 at total GPU budgets 2, 4, and 8.
The current plan does not optimize these ratios using held-out B outcomes.
No claim covers every possible static or hybrid allocation.

## SLO calibration

The calibration set contains 24 distinct ShareGPT conversation prompts.
It is disjoint from the 40-prompt qualification set and the 200-prompt W1 set.
Only DP_Mixed and PD_G1 provide SLO calibration data.
Their low offered rates are 0.3 requests/s for Ouro and 0.08 requests/s for Huginn.

For each model, use `ceil(1.25 * max baseline p90 TTFT)` and `ceil(1.25 * max baseline p90 request TPOT)`.
The controller writes absolute thresholds to `configs/slo_lock.json` before any held-out W1 B run.
It reuses that immutable lock after a resume. It never retunes thresholds against B outcomes.
The secondary strict-streaming threshold uses the same formula on baseline p90 observed ITL.
That secondary metric does not select the primary winner.

## Prospective rate and statistical plan

The first two-GPU ladders are Ouro 1/2/4/8 requests/s and Huginn 0.2/0.4/0.8/1.6 requests/s.
W1 uses 200 distinct conversations, output safety cap128, and natural EOS.
All arms share each seed's prompt order and Poisson arrival offsets.
The base seed is 2026103001. Confirmation seeds add 1000 and 2000.
Run order rotates. Independent confirmations cross physical GPU pairs.
Candidate and control source snapshots remain available per attempt.

Shortlist rates receive two additional independent starts when the budget permits.
The controller preserves every attempted rate and failed arm.
Wilson intervals describe request attainment. They are not independent-run or selection-adjusted capacity intervals.
Final paired intervals must respect trace/run clustering and the small number of starts.

## Queue and denominator contract

The evaluation window ends one mean interarrival interval after the last scheduled arrival.
All planned arrivals remain in the denominator. Failed or incomplete jobs cannot establish capacity.
Qualified-window goodput and drain-inclusive goodput are separate columns.
Observed callbacks define ITL. Unknown timestamps inside coalesced outputs remain unknown.
Successful one-token responses have TPOT0 by the specification's convention.

The backlog is outstanding requests, including running requests. It is not a pure waiting queue.
The exploratory queue check uses samples from the final 75% of the arrival window.
Its slope must not exceed 5% of offered rate.
Its late-minus-early mean must not exceed max(3 requests, 10% of one quarter's expected arrivals).
Windows shorter than 300s cannot receive a stationarity claim from this check.
Passing this heuristic alone does not prove an infinite-horizon sustainable maximum.

## Safety and clock

The wall clock starts at 2026-10-29 21:29:30 KST, before setup.
The hard stop is 2026-10-30 12:29:30 KST.
GPU work stops by 11:44:30 KST to reserve 45 minutes for reporting and publication.
Each job also has a whole-process watchdog and an isolated process session.
Only verified owned burner sessions can stop. Other user work remains untouched.
Correctness jobs can use four independent GPU pairs concurrently.
Clean serving uses a quiet lane with other owned GPU workers and burners stopped.
Idle burners resume outside those timing intervals and at final cleanup.
