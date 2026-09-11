# 1B — Iteration-aware rank-complementary batching

## Decision

**NO-GO.** Per-request rank vectors are predictable but almost collinear, so
there is little complementarity to schedule. The best offline bounded search
reduces assignment makespan by 1.16% and projects 0.17% optimistic E2E.

## Fixed-budget experiment

Every active denoising iteration uses batches of four requests. FCFS, 200 random
orders, perfect-future route knowledge, previous-iteration prediction, and an
EMA predictor all use identical requests, active-position work, and batch size.
The reported `perfect` policy is a strong offline search—100 random greedy
starts plus bounded pair-swap descent—not a proof of the global combinatorial
optimum.

| Policy | Assignment cost | Improvement vs FCFS | Optimistic E2E |
|---|---:|---:|---:|
| FCFS | 1,161,670.0 | — | — |
| Random median | 1,159,989.5 | 0.145% | 0.021% |
| Offline future-aware search | 1,148,188.0 | **1.161%** | **0.166%** |
| Current iteration | 1,158,006.0 | 0.315% | 0.045% |
| EMA | 1,157,716.0 | 0.340% | 0.049% |

Current and EMA recover only 27.18% and 29.33% of the bounded offline gain,
below the requested 50–70% causal gate.

## Diversity gate

- Pairwise aggregate rank-vector cosine: p10 0.9986, p50 0.9995, p90 0.9999.
- 29/30 requests have rank 3 as their aggregate critical rank; one has rank 2.
- Temporal next-rank cosine is excellent (t+1 0.9967), but predictability cannot
  create complementary vectors that are absent from the queue.

The result is therefore not “the predictor is bad.” It is “the physical load
direction is shared by nearly every request.”

## Absolute kill bound

To remove dependence on the heuristic optimizer, a second bound discards both
request indivisibility and batching constraints and fractionally equalizes every
iteration/layer's aggregate assignments across all four ranks. Its assignment
cost is 970,752, a 16.43% reduction from FCFS. Even this impossible relaxation
projects only **2.3557% request E2E**, because the expert stage is 14.33% of
clean request time. The measured load-to-latency calibration projects 0%.

Poisson/bursty queue simulation, aging tuning, and a learned predictor were not
continued: queue delay cannot rescue a mechanism whose zero-queue absolute E2E
upper bound is below the 5% NO-GO threshold. No production scheduler was built.
