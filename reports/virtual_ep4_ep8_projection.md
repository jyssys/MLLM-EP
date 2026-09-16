# LLaDA2.0-mini virtual EP4/EP8 projection

## Status

`VIRTUAL_EP4: READY_FOR_SCREENING`

`VIRTUAL_EP8: READY_FOR_SCREENING`

Labels used in artifacts:

- `SIMULATED-EP4-EP2-CALIBRATED`
- `SIMULATED-EP8-EP2-CALIBRATED`

These are routed-MoE-stage simulations, not measured EP4/EP8 runs. No process
world larger than 2 was created.

## Projection contract

The parallel family remains `DP=1, TP=1, SP=1, EP=P`. Expert ownership is
contiguous:

| Target | Experts/rank | Ownership |
|---|---:|---|
| EP2 | 128 | measured true EP2 |
| EP4 | 64 | virtual contiguous map |
| EP8 | 32 | virtual contiguous map |

EP2 measured equal contiguous source shards. Virtual EP4/EP8 use its balanced
contiguous extension; for non-divisible row counts the first remainder ranks
receive one extra row. This source extension is simulated, not measured dInfer
behavior.

Each prediction combines:

1. actual official-HF refinement/router traces;
2. virtual source and expert-owner mapping;
3. single-H100 replay of each target rank's actual expert histogram using the
   full destination-local grouped MLP path;
4. true-EP2 DeepEP endpoint calibration;
5. a sequential dispatch→critical-rank-expert→combine event model matching the
   current DeepEP normal dependency boundary.

The replicated-state bridge all-gather is absent from every projected EP
number. It is neither EP dispatch nor EP combine.

## Structural projection

The traced cohort contains 5,225 routed-MoE invocations from four complete
GSM8K trajectories.

| Metric | EP2 | SIMULATED-EP4 | SIMULATED-EP8 |
|---|---:|---:|---:|
| Mean max/mean rank load | 1.131 | 1.358 | 1.831 |
| Mean rank-load CV | 0.131 | 0.257 | 0.436 |
| Logical remote dispatch bytes | 32.90 GB | 77.92 GB | 113.98 GB |
| Logical remote combine bytes | 32.90 GB | 77.92 GB | 113.98 GB |

Increasing EP degree lowers per-rank expert compute but increases remote
activation traffic and rank imbalance. Per-invocation CSV files retain the
full assignment matrix `A[s,d]`, deduplicated activation matrix `U[s,d]`, rank
load vector, endpoint byte vector, fanout, and critical rank proxies.

## Calibrated routed-MoE timing

Base-scenario sum over the same 5,225 trace invocations:

| Component | EP2 (ms) | SIMULATED-EP4 (ms) | SIMULATED-EP8 (ms) |
|---|---:|---:|---:|
| EP dispatch | 785.7 | 854.5 | 847.4 |
| Critical routed expert compute | 5,490.9 | 4,199.7 | 3,504.1 |
| EP combine | 428.0 | 471.2 | 466.8 |
| Routed-MoE stage | **6,704.6** | **5,525.5** | **4,818.2** |

Relative to the calibrated EP2 MoE-stage projection, this corresponds to:

- SIMULATED-EP4: 17.59% lower routed-MoE-stage time;
- SIMULATED-EP8: 28.14% lower routed-MoE-stage time.

These are not full-forward or BCT speedups. Attention, dense/shared paths,
controller overhead, and the replicated-state bridge are excluded.

## Scenario sensitivity

| Scenario | SIMULATED-EP4 total (ms) | SIMULATED-EP8 total (ms) |
|---|---:|---:|
| endpoint optimistic (EP2 P10 fit) | 5,463.7 | 4,757.3 |
| EP2-calibrated base (median fit) | 5,525.5 | 4,818.2 |
| concurrency stressed (EP2 P90 fit) | 5,720.5 | 5,012.6 |

Scenario ranges are sensitivity envelopes, not statistical confidence
intervals. They capture measured EP2 latency variation but cannot identify
true 4-way/8-way endpoint concurrency. Consequently EP8 has materially higher
transfer risk than EP4.

The nearest measured compute-envelope distance has median/P90:

- EP4: 0.175 / 0.276;
- EP8: 0.167 / 0.273.

No compute time is obtained by dividing EP2 by P. Each target uses an actual
target-rank replay.

## Refinement behavior

This is a vanilla, non-Epoch baseline. Within a block, accepted/MASK state can
change dramatically while all prefix/current-block rows continue through every
MoE layer. Therefore refinement progression changes routing, ownership load,
fanout, and bytes but does not manufacture a fresh-row decline.

Required plots are under
`artifacts/virtual_ep/20260916_215010/ep4_prediction/figures/`:

- remaining MASK vs iteration;
- accepted tokens vs iteration;
- EP4 max/mean rank load vs iteration;
- EP4 remote bytes vs iteration;
- EP4 fanout vs iteration;
- EP4 dispatch/expert/combine prediction vs iteration;
- EP4 layer×iteration imbalance heatmap.

Git-tracked copies for review are in `reports/figures/virtual_ep4/`. The
machine-readable headline values and accounting boundaries are in
`reports/virtual_ep_simulator_summary.json`; the full per-invocation replay
tables remain in the local artifact tree.

## What is measured

- official model routes, router weights, mask/acceptance state, and NFE;
- true EP2 ownership and cross-rank dispatch/combine correctness;
- DeepEP normal dispatch/combine latency on GPU0↔GPU1;
- single-H100 target-rank BF16 grouped expert replay for EP2/EP4/EP8 shapes;
- held-out true-EP2 routed-MoE replay latency;
- EP-isolation bridge all-gather only as a separate diagnostic category.

## What is simulated

- EP4/EP8 source-rank mapping;
- EP4/EP8 expert ownership and rank-local load;
- source-destination assignment and unique-activation traffic;
- endpoint-limited dispatch/combine latency;
- critical-rank expert time and routed-MoE stage critical path.

## What still requires true EP4/EP8

- multi-peer NVSwitch/DeepEP endpoint contention and control overhead;
- exact source partition/padding behavior in a production runtime;
- physical protocol bytes and hardware traffic counters;
- backend-specific overlap at four/eight ranks;
- numerical trajectory/quality at EP4/EP8;
- full-forward, BCT, throughput, queueing, and tail latency.

## CLI

Structural projection:

```bash
python -m virtual_ep.simulate \
  --trace artifacts/virtual_ep/<run>/trace/<trace>.npz \
  --ep 4 \
  --structural-only \
  --output artifacts/virtual_ep/<run>/ep4_prediction/structural
```

Calibrated timing projection:

```bash
python -m virtual_ep.simulate \
  --trace artifacts/virtual_ep/<run>/trace/<trace>.npz \
  --ep 4 \
  --scenario ep2_calibrated_base \
  --communication-model artifacts/virtual_ep/<run>/communication/model.json \
  --compute-model artifacts/virtual_ep/<run>/compute_model/ep4_grouped_mm.csv \
  --output artifacts/virtual_ep/<run>/ep4_prediction/ep2_calibrated_base
```

Supported targets are `--ep 2|4|8`; supported scenarios are
`endpoint_optimistic|ep2_calibrated_base|concurrency_stressed`.

## Phase-3 protocol (not executed)

Every future threshold/controller candidate must first run a fresh real
single-GPU rollout to obtain its own accuracy, NFE, refinement state, and
routes. A threshold-0.95 trace must never be re-thresholded offline to invent a
different future trajectory. After virtual screening, baseline, conservative,
and aggressive representatives should be rerun through true EP2 to check
ranking drift. True EP4 remains the primary final validation target.

## Recommendation

Use SIMULATED-EP4 for candidate screening now, but reserve headline latency and
quality claims for a future true EP4 run. EP8 can be used as secondary
structural/scaling sensitivity only until multi-peer calibration is available.
