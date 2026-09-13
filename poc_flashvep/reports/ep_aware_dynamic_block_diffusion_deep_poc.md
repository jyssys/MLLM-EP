# EP-aware dynamic block diffusion deep PoC

## Executive decision

**CHARACTERIZATION-SIGNAL.** LLaDA2.0-Flash 100B on true routed EP4 shows a
clear block-size-dependent physical execution transition. However, real mixed-B
trajectories do not create material quality-safe request-level headroom over
the strongest fixed-B frontier. The controller gate therefore fails and no live
controller or production scheduler is implemented.

This is not a substrate failure or a one-point negative. The campaign repaired
hidden benchmark overrides, validated B=8/16/32/64/128, controlled endpoint,
early stop, threshold, mini size, and physical M, executed reversed and barrier
schedule controls, tested per-request future-aware schedules, enumerated actual
mixed trajectories, and repeated the result at longer generation lengths.

## Evidence boundary

- Clean measured evidence: BCT, throughput, NFE, bounded GSM8K/HumanEval score,
  output, and peak HBM.
- Observer-heavy measured evidence: routed rows, expert support, rows/expert,
  fanout, rank load, dispatch bytes, and same-device component timing. These
  traces are not used as clean BCT evidence.
- Modeled sensitivity: dead/live row mapping for a hypothetical Epoch-like
  compactor. This is not an Epoch speedup measurement.
- Dynamic evidence: every mixed schedule is actually rolled out. Independent
  per-block timings are never summed into a fake oracle.

## Substrate and topology truth

Every task launch used `CUDA_VISIBLE_DEVICES=4,5,6,7`, mapping task-local ranks
0--3 to physical GPUs 4--7. The downloaded checkpoint has 32 layers, hidden
size 4096, BF16 weights, 256 routed experts with top-k 8, one shared expert,
and MoE intermediate size 1024. The production-like path is dense TP4 plus
routed EP4, DP1/PP1. Each rank owns 64 contiguous routed experts; DeepEP sends
remote rows to the owner-rank fused expert path and reverse-combines outputs.
UUID, topology, ownership, remote receive, and backend evidence are retained in
the workspace.

## Variable-block correctness

The stock config-42 benchmark silently forced B=32 and threshold=0.9 and used a
B32-aligned terminal bucket. The isolated dInfer patch removes those overrides,
adds a common absolute target length, and supports block-boundary schedules
while preserving completed-prefix visibility, within-block bidirectionality,
future masking, absolute positions, KV writeback, and EOS behavior.

Fixed `[32]` and `[64]` schedules match the ordinary fixed paths exactly on
answer strings and generated lengths for 32/32 samples on both tasks. A
threshold-1.0, no-early-stop control still changes NFE across B, proving that B
changes the decoding trajectory; request latency cannot be interpreted as a
pure EP microbenchmark. Matched-M traces separately isolate the physical effect.

## Strong fixed-B frontier

At submitted batch 32, the reproducible key points are:

| Task | B / mini / threshold | median BCT | score | restarts |
|---|---|---:|---:|---:|
| GSM8K | 32 / 32 / 0.9 | 9.453 s | 13/32 | 3 |
| GSM8K | 64 / 16 / 0.9 | 11.725 s | 14/32 | 3 |
| HumanEval | 32 / 16 / 0.85 | 13.881 s | 18/32 | 3 |
| HumanEval | 8 / 16 / 0.85 | 20.167 s | 19/32 | 3 |

B64/mini16/t0.85 reaches 18/32 HumanEval in 13.613 s in one launch, so B64 is
also retained on the frontier rather than discarded. HumanEval mini32 exceeds
the validated memory/source-row envelope and is reported as capacity-blocked.

## AR-ness × EP physical regime

At fixed n=8/mini8 on GSM8K, B8 to B128 changes median physical rows from 48 to
704, active experts from 101.9 to 191.9, rows per active expert from 3.67 to
29.76, and BF16 dispatch bytes from 2.36 MB to 34.90 MB. The fraction of active
experts with at most four rows falls from 75.4% to 29.3%. Critical dispatch rises
from 1.335 ms to 9.145 ms, while critical fused-expert time rises only from
0.536 ms to 1.005 ms. HumanEval has the same direction.

Thus small B is startup/tiny-GEMM dominated; larger B substantially improves
expert packing but grows payload, dense-row work, and modeled dead work. Mean
fanout stays near 3 and remote fraction near 0.75, so the transition is not a
new rank-topology regime. It is principally a physical-M, packing, payload, and
liveness regime.

The matched configured-M=256 control gives median per-wave walls of 117.83,
125.59, and 122.01 ms for B16/mini16, B32/mini8, and B64/mini4. These points
converge within about 7%; B128/mini2 remains slower at 172.75 ms due to context
and scheduling work. Full-request cost does not converge because the number of
physical waves grows sharply as mini shrinks. This is direct evidence that B
changes physical shape, while also showing that best-per-B batching absorbs
most of the supposed EP-specific opportunity.

Measured decision-live state maps to approximately 50--62% dead rows depending
on B/task. Removing those rows belongs to Epoch-like compaction and is not
credited to dynamic B. The earlier same-substrate RAWS sensitivity found only
0.650% ready-set dynamic-mini headroom and 0.224% after hypothetical compaction;
these are contextual controls, not measurements of Epoch.

## Request heterogeneity and schedule economics

An isolated mini=1 future-aware tournament finds real request heterogeneity:
GSM8K preferred fixed B histogram is B8:1, B32:4, B64:1, B128:2; HumanEval is
B16:1, B32:6, B64:1. Yet median quality-safe mixed-schedule gain is only 1.014%
and 0.369%, respectively. Two GSM8K requests reach about 9.15% and 10.26%, but
the aggregate median fails the 5% gate and mini=1 sacrifices batching.

In batched execution, switching width creates a real fragmentation tax: requests
finish blocks at different refinement iterations, split into width-specific
physical waves, and often increase NFE. Reversing policy order and forcing a
global schedule barrier do not recover the loss. Longer-generation controls also
leave every tested mixed schedule far behind its quality-matched fixed endpoint.

The decisive search executes all 56 ordered compositions of 128 using B in
`{16,32,64,128}` on each task. GSM8K fixed B32 takes 8.200 s at 5/8, while the
fastest equal-score mixed schedule takes 10.596 s: **-29.22%** gain. HumanEval
fixed B128 takes 7.576 s at 3/8, while the fastest equal-score mixed schedule
takes 11.447 s: **-51.10%** gain. O0 latency-only, O1 score-preserving, O2
per-request-safe, and O3 one-sample-epsilon all select these same negative mixed
candidates. Higher-score mixed schedules exist on the bounded samples, but are
slower still.

## Trigger and method gate

Physical M, rows/expert, tiny-expert share, and dispatch bytes explain the
per-wave execution regime. Fanout and remote fraction add little because they
barely move. None of these EP observables resolves the semantic NFE/quality
effect that determines the request-level safe winner. Since the future-aware
target itself is below the gate, fitting a learned predictor would manufacture
an accuracy metric for noise.

Accordingly:

- semantic-only variable block sizing is already occupied by AdaBlock-dLLM and
  DSB and is not relabeled here;
- EP-only static shape tuning is substantially recovered by best-per-B mini
  calibration;
- a joint EP+semantic live controller has no material oracle left to recover.

No live controller, TP4 follow-up, or online-serving speedup claim is made.

## Prior-art boundary

AdaBlock-dLLM chooses block size from delimiter/confidence semantics; DSB uses
semantic difficulty and sliding boundaries; Adaptive Block Diffusion trains
over varying prefix-window configurations; Block Diffusion establishes B as an
AR-ness axis. Epoch removes dead routed rows. dInfer and dLLM serving runtimes
provide global block scheduling and continuous batching. Therefore changing B,
semantic confidence triggering, and dead-row removal are not novel.

The only potentially distinct systems claim would have been that measured EP
state supplies extra predictive value and that a dynamic trajectory beats all
fixed-B Pareto points. This campaign verifies the physical premise but falsifies
the economic premise, so it does not promote a method.

## Robustness

The primary fixed points use three independent restarts. Schedule order was
reversed, a barrier variant was tested, and longer-generation B16/B32/B64/B128
controls were run on both tasks. Observer-heavy trace wall tax has a median near
50.1% on GSM8K and 37.3% on HumanEval, so those traces are never used as clean
latency claims. Failed setup and capacity runs remain in `ATTEMPT_LOG.csv` rather
than being silently removed.

## Answers to the required questions

1. **Does B change physical EP work?** Yes: rows, pairs, expert support,
   rows/expert, bytes, and component latency all move materially.
2. **Is small B startup/tiny-GEMM dominated?** Yes. At B8, 75.4% of active
   experts have at most four rows in the GSM8K control.
3. **Is large B communication/fanout/liveness dominated?** Payload and modeled
   liveness waste grow strongly; fanout does not. Calling it fanout-dominated
   would be unsupported.
4. **Is there an AR-ness/EP regime transition?** Yes, from tiny/startup-heavy to
   better-packed but larger-payload/dense-row waves.
5. **Is this only steps/NFE?** No for physical shape, as matched-M replay shows;
   NFE/trajectory remains an inseparable part of request-level behavior.
6. **Does it survive matched M?** Partly. B16--64 converge within about 7%,
   while B128 retains extra context/scheduling cost.
7. **What fixed B wins?** It depends on task, quality target, and request. B32
   is the strongest reproducible GSM8K point; B32/B64 form the main HumanEval
   speed frontier, while B8 buys one extra bounded HumanEval success at high cost.
8. **Is official B32 Pareto-optimal?** Yes, though it is not the only Pareto
   point.
9. **Do requests prefer different safe B?** Yes, but median exploitable
   future-aware gain is only 1.014% on GSM8K and 0.369% on HumanEval.
10. **Maximum latency-only dynamic oracle?** The exhaustive batched result is
    negative: -29.22% GSM8K and -51.10% HumanEval. The maximally favorable
    isolated-request median is only 1.014%.
11. **Zero-quality-drop oracle?** O1 and O2 select the same negative exhaustive
    candidates; no positive aggregate zero-drop schedule exists.
12. **Small-epsilon oracle?** Permitting one bounded-sample loss does not
    produce a faster mixed schedule.
13. **Does the oracle require future information?** Even the mini=1 positive
    tail does. A live controller has less information and also pays batching cost.
14. **Which EP features predict B?** M, rows/expert, tiny-expert share, and bytes
    predict wave cost, not the quality-safe next trajectory.
15. **Do EP features add value beyond semantics?** No material request-level
    incremental value is demonstrated.
16. **Can a simple live policy recover useful gain?** No; the perfect target is
    already below the implementation gate.
17. **Does the gain reproduce at longer generation?** The result reproduces as
    a negative: mixed schedules remain substantially slower.
18. **Does it reproduce on both tasks?** Yes.
19. **Is a method distinct from AdaBlock established?** No. The possible EP
    distinction has no economic oracle.
20. **Is this paper-level systems headroom?** No. The physical characterization
    is useful, but dynamic-B scheduling should be closed on this substrate.

## Artifacts

- Working reports and machine-readable results: `poc_ep_dynamic_block/`
- Runtime patch: `poc_ep_dynamic_block/dinfer_dynamic_block.patch`
- Reproduction instructions: `poc_ep_dynamic_block/REPRODUCE.md`
- Detailed decision: `poc_ep_dynamic_block/reports/final_decision.md`
