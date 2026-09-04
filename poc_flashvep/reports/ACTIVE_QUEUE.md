# Live hypothesis queue — autonomous discovery loop

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
