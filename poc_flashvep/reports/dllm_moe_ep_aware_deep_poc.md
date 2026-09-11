# Training-Free EP-Aware Inference for MoE Diffusion LMs: Deep PoC

## Executive decision

**Final label: CHARACTERIZATION-ONLY.**

The study establishes one strong causal characterization: TEAM's released
training-free policy is beneficial on one GPU, weakly beneficial on true EP2,
and slower than vanilla on true EP4.  The reason is an algorithmic/physical
mismatch: fewer denoising forwards coexist with about 32% more token-expert
assignments and remote bytes, while destination fanout remains saturated.

That finding does not support a new method yet.  The strongest request-level
perfect oracle among six EP-aware action policies is only 6.28%, below the 8%
promotion gate.  The most implementable policy, TEAM EP-aware speculation
width, has a generous 4.01% upper bound and falls to 2.01% if communication
becomes 2× faster.  REFLEX and DES ports failed bounded quality checks and are
used only as structural diagnostics, not method performance evidence.

## Runtime and correctness

All new GPU jobs used physical GPUs 0–3.  The EP4 TEAM path is TP1/DP1/EP4:
128 global experts, 32 resident experts per rank, exact NCCL assignment A2A,
vLLM fused local experts, and reverse A2A combine.  Its layer reference check
gave cosine 0.99999994 and relative L2 0.0560%.  This is true physical EP but a
reference transport rather than production DeepEP.

The LLaDA diagnostic similarly owns 16/64 experts per EP4 rank and transports
ragged assignments.  Stock vLLM 0.10.2 naive EP instead multicasts full hidden
states and router logits and all-reduces full output; selected k does not alter
its communication bytes.

## Single → EP2 → EP4 characterization

| Method | Model | Setting | Quality status | Speedup vs own vanilla | Native proxy | Expert pairs | Remote work | Fanout |
|---|---|---|---|---:|---:|---:|---:|---:|
| TEAM | SDAR-30B-A3B | single | bounded positive | **1.832×** | NFE -41.7% | +11.5% | N/A | N/A |
| TEAM | SDAR-30B-A3B | EP2 | trajectory preserved | **1.179×** | NFE -33.3% | +31.5% | assignments/bytes +33.6% | 2→2 |
| TEAM | SDAR-30B-A3B | EP4 | semantic prefix only | **0.821× median** | NFE -21.4% | +32.1% | assignments/bytes +32.1% | 4→4 |
| REFLEX port | LLaDA-MoE-7B | single | **invalid: 0/4 vs 3/4** | excluded | AvgK -6.05% | -6.05% | N/A | N/A |
| REFLEX structural | LLaDA-MoE-7B | EP4 | invalid | excluded | AvgK -6.05% | -6.05% | ragged rows -7.07%; stock bytes 0% | 4→4 |
| DES port | LLaDA-MoE-7B | single | **invalid: 1/4 vs 3/4** | excluded | coreset -40.6% | top-8 unchanged | approximately unchanged | 4→4 |

TEAM EP4 clean paired restart speedups were 0.902×, 0.716×, and 0.821×;
the median is 0.821×.  A fourth attempt hung in NCCL after its baseline and
was excluded.  EP4 detailed stage tracing is observer-heavy: it supports work
localization but not clean latency claims.

## Algorithmic proxy versus physical EP cost

- **TEAM NFE is the clearest mismatch.** Calls fall while physical branch rows
  and bytes rise by roughly one third.  Fewer active experts reduce local
  expert time, but larger fully fanned-out forwards erase the gain.
- **REFLEX AvgK is backend-conditional.** Six percent fewer expert pairs gives
  7.07% fewer remote rows in ragged A2A, but zero byte reduction in stock naive
  EP and no fanout change.
- **DES unique experts is not fanout.** A 38/64 coreset still spans all four
  ranks with top-8 routing.

Only TEAM has a quality-valid end-to-end result, so the spec's two-method
generality gate fails.

## EP4 cost surface

The 84-shape calibration uses 256–4,096 assignments, fanout 1–4, remote
fraction 0.25–0.75, and 2/8/16 active experts/rank.  Median total operator time
is 0.6127/0.6089/0.6384 ms at 256/1,024/4,096 assignments.  A 16× work-count
change therefore moves time by only about 4% in the startup-dominated range.

The best held-out model adds critical-rank features: RMSE 0.0491 ms, MAPE
4.44%, R² 0.058.  Low MAPE comes from the flat floor; near-zero R² makes the
model unsuitable for choosing actions.

## Candidate competition

| Candidate | Perfect request E2E oracle | Decision |
|---|---:|---|
| A. Cost-aware re-ranking | 4.01% | kill |
| B. EP-budgeted work shaping | 3.83% | kill |
| C. Marginal utility / EP cost | 4.01% | kill |
| D. Topology-aware selection | 3.90% | kill |
| E. TEAM EP-aware speculation width | 4.01% | kill |
| F. Perfect-future joint decode–EP action | **6.28%** | weak, no prototype |
| All EP communication disappears | 15.60% | impossible ceiling, not a candidate |

Candidate E already assumes optional speculative communication can disappear
without increasing NFE or reducing acceptance.  Its bound is below 5% before
policy overhead, and static width tuning is a trivial-fix attack.  Candidate F
uses future information and still remains below 8%.

No independent attention/speculation/communication overlap window above 5%
was demonstrated.  The per-request router→dispatch→expert→combine chain is
strict, and moving work changes the TEAM trajectory.

## Quality boundary

The official TEAM positive trend was reproduced: bounded GSM8K was 3/4 versus
3/4, and the 256-token boundary pair was 2/2 versus 2/2.  TEAM EP4 timing uses
short generation and cannot claim benchmark accuracy beyond semantic prefix
agreement.  REFLEX and DES results are explicitly `PORT/EXECUTION-CONTRACT_FAILURE`,
not negative results about the papers.

## Prior-art conclusion

Epoch accelerates the execution of already-required fresh work; the intended
candidate would decide which optional dLLM work to create using physical EP
cost.  That conceptual distinction exists, but it is insufficient without
headroom.  REFLEX and DES already expose the relevant optional-action spaces,
making bytes/fanout-aware scoring an adjacent extension.  DICE also makes a
generic overlap framing unsafe.  No exact paper-worthy gap survives both the
novelty and economic gates.

## Answers to the ten required questions

1. TEAM scales 1.832× → 1.179× → 0.821×; REFLEX/DES scaling is not
   quality-valid and is not claimed.
2. TEAM NFE is the worst physical-cost proxy.  Pair count is the most directly
   related to ragged bytes, but only conditionally and without quality-valid
   REFLEX evidence.
3. TEAM's amplification remains about 32–34%; its performance consequence
   worsens on EP4 enough to reverse the winner.
4. REFLEX rows fall 7.07% in ragged A2A, but stock naive bytes fall 0%.
5. DES's active coreset falls 40.6%; EP4 fanout falls 0%.
6. The common cost model is not decision-grade (best R²=0.058).
7. No candidate using it reaches the implementation gate.
8. Candidate F is largest at 6.28%; Candidate E is implementable but only 4.01%.
9. At 2× faster communication Candidate E shrinks to 2.01%.
10. There is not enough evidence for a paper method or EP8 validation.

## Final recommendation

Do not implement a production replica manager, custom communication kernel,
dynamic-k DeepEP runtime, or training-free EP-aware controller from this PoC.
Keep TEAM's EP4 reversal as a useful reproducible characterization and backend
lesson.  Reopen only with a faithful quality-valid REFLEX/DES substrate or a
new optional action family whose direct quality-matched oracle exceeds 8%,
preferably 12%.

Detailed reports and raw analysis are under `poc_dllm_ep_aware/`; result root:
`poc_dllm_ep_aware/results/deep_20260912_030000/`.
