# Live hypothesis queue — autonomous online-runtime discovery (2026-09-05)

The queue is intentionally kept at eight or more pending questions.  Statuses
are updated after each bounded live measurement; a negative result generates at
least two follow-ups rather than ending the loop.

| ID | Hypothesis | Evidence so far | Expected signal | Required control | Estimated runtime | Novelty risk | Priority | Status |
|---|---|---|---|---|---:|---|---:|---|
| H1b | More active experts can improve grouped-GEMM concurrency at fixed work | Qwen3-VL A2→A32 expert −35.7% at M512, −22.7% M1024; A32 extension | crossover vs M/A ceiling | exact M/rank load/hidden and real-route check | 60–90 min | medium | 1 | PROMISING |
| H1c | Active-expert sweet spot changes at a kernel-regime boundary | First 5-rep grid suggested collapse near M4096, but 100-rep repeat did not reproduce A32 gain | non-monotonic A curve after state-controlled repeat | 5+ reps per point, kernel identity | 45 min | medium | 2 | HOLD |
| H1d | Kernel selection changes explain inverse fragmentation | H1 reversal | discontinuity at A/M | same shape with kernel-name capture | 30 min | medium | 3 | PENDING |
| H1e | Natural routes occupy the fast fragmentation band | prior real routes, controlled H1 | real-vs-controlled gap | matched assignment/rank-load replay | 45 min | low | 4 | PENDING |
| H5b | Temporal warmth survives randomized persistent-worker order | 120 interleaved targets: all condition deltas within 1.4% | >5% target-B delta | 30 target measurements/condition, randomized order | 90–120 min | medium | 1 | REJECTED |
| H5c | The H5 tail is case-position/allocator state, not route history | randomized run position-duration Pearson −0.443, CV up to 47.5% | tail follows position | repeat same route under shuffled order | 30 min | low | 2 | PROMISING |
| H15b | Layer×M piecewise model removes residual clusters | H15 R²=.326, 37 large residuals | residual reduction >10% | pre-registered bins, held-out requests | 45 min | high | 3 | PENDING |
| H6b | EP fanout affects communication after volume/rank load matching | F1→F4 expert −12.4…−17.1% at M512; +104…+114% at M128 across layers4/24/44 | phase-specific fanout×M regime | fanout-controlled same-volume routes | 60 min | medium | 5 | PROMISING |
| H10b | Expert weight norm predicts layer cost at fixed route shape | layer spread at M512 | weight-stat/cost relation | same route shape across layers | 45 min | medium | 6 | PENDING |
| H13b | Visual category effect persists after layer/kernel matching | category spread 30% at M512 | >=5% paired category delta | same M/layer/repetition | 45 min | high | 7 | PENDING |
| H16 | Persistent worker route replay has a deterministic first-shape warmup/state effect | A2/A32 same-M order flips A32 effect from +60.8% to −37.3% expert; wall +134.6% to −54.0% | first-use/allocator/kernel state explains sign flip | same routes, two order permutations, 50 reps | 30 min | low | 1 | STRONG_DIAGNOSTIC |
| H17 | M=512 triggers a layer-specific launch regime | H10/H15 residuals | discontinuity by layer | independent per-layer repetitions | 30 min | medium | 2 | GENERATED |
| H18 | Driver metadata can disagree with physical binding | stale trigger field observed | automated PID/device mismatch | nvidia-smi + worker proof | 15 min | low | 8 | GENERATED |
| H19 | Fanout changes dispatch/combine, not only expert GEMM | H6 phase medians differ by M/fanout | dispatch/combine-specific crossover | paired F1/F2/F4, same assignment load | 30 min | medium | 3 | ACTIVE |
| H20 | A32 fast band is caused by grouped-GEMM launch geometry rather than routing semantics | A32 improves both Qwen3-VL and Qwen3 at small/medium M | kernel-name / launch-shape discontinuity | one bounded kernel trace or backend metadata capture | 30 min | medium | 4 | PENDING |
| H21 | Natural Qwen3-VL routes fall near the A32/fanout fast band | controlled A32 benefit is large | natural route geometry proximity | matched real route replay, no synthetic route changes | 45 min | low | 5 | PENDING |
| H22 | H6 fanout phase effect is driven by expert launch shape while combine remains noisy | M512 expert −12…−17%; combine changes large but unstable | phase-specific launch/communication regime | same routes, longer persistent repetitions | 30 min | medium | 6 | GENERATED |
| H23 | Route-order state can contaminate any shape comparison unless first-use kernels are separately prewarmed | H16 sign flip and H5 position correlation | order-invariant medians after global prewarm | randomized case order with per-shape compile warmup | 30 min | low | 7 | GENERATED |

## Sprint-local runtime hypotheses (seeded from the current specification)

These are intentionally kept pending while measurements are collected; a
negative result generates a follow-up rather than ending the discovery loop.

| ID | Hypothesis | Evidence so far | Expected signal | Required control | Estimated runtime | Novelty risk | Priority | Status |
|---|---|---|---|---|---:|---|---:|---|
| H1R | DeepEP high-throughput communication SM allocation has a workload-dependent optimum | vLLM exposes VLLM_DBO_COMM_SMS and DeepEP Buffer.set_num_sms | >=5% best-per-regime vs static | same requests/routes, per-shape warmup | 35 min | medium | 1 | PENDING |
| H2R | Dispatch and combine prefer different communication-SM settings | source has separate dispatch/combine configs | phase ranking reversal | identical route snapshot | 25 min | medium | 2 | PENDING |
| H3R | DeepEP backend ranking crosses within prefill, not only prefill/decode | deepep HT and LL are available in config | >=5% within prefill | matched M and phase | 40 min | high | 3 | PENDING |
| H4R | Mixed scheduler steps expose a backend/config anomaly after matching M | online trace includes prefill/decode waves | >=5% residual effect | same M, phase composition | 25 min | high | 4 | PENDING |
| H5R | DP-source asymmetry creates an online EP critical-path penalty beyond destination load | prior proxy was indirect | >=5% matched effect | matched global M/rank loads | 30 min | high | 5 | PENDING |
| H6R | Empty/dummy DP participants add a nonlinear communication tax | V1 DP metadata has empty forwards | >=10% at a volume boundary | both DP ranks active vs one empty | 25 min | low | 6 | PENDING |
| H7R | Identical routing shapes have layer-specific runtime configuration rankings | prior layer spread | >=5% ranking change | same M/routes across layers | 30 min | medium | 7 | PENDING |
| H8R | Slow online T_MoE tails are caused by queue/state rather than route/load | previous held-out model R2 near zero | matched tail phase explains >=10% | same M/layer/load fast controls | 30 min | medium | 8 | PENDING |
| H9R | Cross-rank tail co-occurrence distinguishes communication queue stalls from local expert compute | no direct rank timing yet | stable single/global category | rank-wise event proxy | 25 min | medium | 9 | PENDING |
| H10R | Burst and steady workloads with matched average rate have different sustained T_MoE tails | online waves available | >5% after idle warmup control | matched total tokens/rate | 25 min | medium | 10 | PENDING |
| H11R | Safe runtime configurations leave measurable per-regime oracle headroom | only HT currently validated | 3–10% lower envelope | all configs separately restarted | 40 min | high | 11 | PENDING |
| H12R | Generic Qwen3 text-only and Qwen3-VL share the strongest runtime anomaly | prior MLLM trace exists | replication or clear divergence | same shape/config | 35 min | high | 12 | PENDING |
