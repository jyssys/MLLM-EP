# Fair Route-Oracle Chunk Decomposition

**Result:** `FINAL STATUS: NO-GO` for a routing-only chunk-boundary opportunity
under the preregistered fair comparison.  The earlier large GPU reductions are
explained primarily by the relaxed chunk-size/invocation policy, not by the
routing-aware boundary at a fixed invocation count.

## 1. Scope and configuration

This PoC keeps token order, expert IDs, router weights, assignment totals,
expert placement, and dtype fixed.  Only contiguous chunk endpoints change.
No scheduler, router, placement, communication, or kernel implementation was
added.

| item | value |
|---|---|
| base commit | `16df6572710fc881c1a00cd736244ef02878eb66` |
| model | Qwen3-VL-30B-A3B-Instruct, local validated snapshot |
| dtype | BF16 |
| vLLM | 0.20.0, eager |
| parallelism | TP2 / DP2 / EP4 / PP1 |
| communication | DeepEP high-throughput |
| MoE backend | TritonExperts / `DeepEPHTPrepareAndFinalize` |
| placement | linear, 32 of 128 experts per EP rank |
| GPUs | physical **1,2,3,4 only** (`CUDA_VISIBLE_DEVICES=1,2,3,4`) |
| DBO / prefix cache | off / off in the replay trigger |
| short routes | 8 representative real-image routes, 5 route layers (0,12,24,36,47) |
| long routes | 4 local multi-image routes, layer 24 |
| timing | 5 warmups + 20 measured iterations per rank/configuration |
| activation provenance | validated layer-24 hidden-state/weight capture, cycled to route length (same operator-replay protocol as the preceding GPU validation) |

The long route files contain 3,203, 6,984, 5,234, and 9,065 decoder tokens;
the short files span 128--395 tokens.  Offline analysis covers all 12 routes
and all 48 layers.  GPU replay covers 40 short and 4 long request/layer
observations per strategy and budget.

## 2. Strategies and fairness constraints

* **Fixed:** contiguous `[B,B,...,tail]` chunks.
* **Balanced:** exactly `K=ceil(L/B)` chunks, with sizes differing by at most
  one token; no routing information.
* **Same-count Oracle:** exact route-aware DP, exactly the Fixed `K`, every
  chunk `<=B`.
* **Strict Oracle:** exact route-aware DP, variable chunk count, every chunk
  `<=B`.
* **Relaxed Oracle:** prior diagnostic DP with chunk sizes in
  `[0.75B,1.25B]`; it is not a production-feasible main result.

The route-aware DP objective is the preregistered sum of visual expert tile
counts.  Both all-token and visual-only shape metrics are retained.  Every
strategy was checked to partition all tokens exactly once; GPU route identity,
token-partition identity, and output correctness passed for every observation.

The first GPU attempt was retained separately but not used: it measured methods
in a fixed order and showed a first-call warm-cache bias even when the groups
were identical.  The final run prewarmed all five strategies and used a
deterministic per-observation strategy permutation.  Only that corrected run
is used below.

## 3. Offline decomposition

Reduction is `1 - strategy/fixed` for the tile proxy.  Values are paired
medians across the same request/layer observations.

| budget | Fixed chunks | Balanced chunks | Same-count chunks | Strict chunks | Relaxed chunks | Fixed→Balanced | Balanced→Same-count | Same-count→Strict | Strict→Relaxed | Fixed→Relaxed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|128|11.5|11.5|11.5|13.0|9.5|−0.27%|+0.27%|−1.66%|+27.39%|+16.09%|
|256|6.0|6.0|6.0|7.5|5.0|0.00%|0.00%|−2.24%|+20.86%|+14.17%|
|512|3.0|3.0|3.0|3.0|3.0|0.00%|0.00%|0.00%|+5.47%|+5.47%|
|1024|1.5|1.5|1.5|2.0|1.5|0.00%|0.00%|0.00%|0.00%|0.00%|

The offline visual-tile proxy gives the same conclusion: routing-only
Balanced→Same-count is +0.00--0.73% at 128/256 and zero at the larger short
scale.  The prior relaxed headroom is mostly a chunk-count/size relaxation;
it is not a fair estimate of a routing-only boundary effect.

## 4. GPU replay results

The primary metric is max-rank expert CUDA time (the median of each rank's
per-configuration median).  Values are milliseconds; reductions are relative
to Fixed.

### Short representative routes

| budget | Fixed | Balanced | Same-count Oracle | Strict Oracle | Relaxed Oracle |
|---:|---:|---:|---:|---:|---:|
|128|1.884|2.157 (−14.5%)|2.038 (−8.2%)|2.515 (−33.5%)|1.440 (+23.6%)|
|256|1.078|1.176 (−9.1%)|1.078 (+0.1%)|1.564 (−45.0%)|0.887 (+17.7%)|
|512|0.640|0.639 (+0.1%)|0.640 (−0.0%)|0.641 (−0.3%)|0.640 (−0.1%)|
|1024|0.636|0.636 (−0.1%)|0.637 (−0.1%)|0.637 (−0.1%)|0.637 (−0.1%)|

### Long multi-image routes

| budget | Fixed | Balanced | Same-count Oracle | Strict Oracle | Relaxed Oracle |
|---:|---:|---:|---:|---:|---:|
|128|18.297|18.232 (+0.1%)|18.207 (+0.1%)|19.809 (−8.3%)|15.276 (+16.5%)|
|256|10.327|10.336 (−0.1%)|10.311 (+0.2%)|11.335 (−9.8%)|8.472 (+18.0%)|
|512|6.195|6.327 (−2.1%)|6.177 (+0.3%)|6.397 (−3.3%)|5.230 (+15.6%)|
|1024|4.167|4.066 (+2.4%)|4.439 (−6.5%)|4.785 (−14.8%)|4.366 (−4.8%)|

At the required small-budget short set, the routing-only GPU reduction is
**−0.06% (128)** and **+0.33% (256)**.  At 512/1024 it is **+0.10%** and
**−0.05%**.  The long-route values are +0.04%, +0.26%, +0.07%, and −0.52%
for 128/256/512/1024.  None approaches the 5% HOLD threshold.

The corresponding short wall-time routing-only reductions are −0.08%, +0.05%,
+0.03%, and +0.02%; dispatch and combine changes are small and inconsistent.
Thus the absence of an expert-kernel gain is not hidden by an opposing
communication component.

## 5. Component decomposition

The paired GPU expert-time decomposition is:

| source | budget | tail balancing (Fixed→Balanced) | routing-only (Balanced→Same-count) | strict chunk-count flexibility | relaxed >B/fewer-invocation effect | total Fixed→Relaxed |
|---|---:|---:|---:|---:|---:|---:|
|short|128|+0.03%|**−0.06%**|−24.59%|+29.06%|+13.41%|
|short|256|−0.16%|**+0.33%**|−9.98%|+21.24%|+6.04%|
|short|512|+0.00%|+0.10%|−0.15%|+0.05%|+0.25%|
|short|1024|−0.14%|−0.05%|+0.01%|−0.05%|−0.05%|
|long|128|+0.14%|+0.04%|−7.44%|+22.29%|+16.13%|
|long|256|−0.06%|+0.26%|−14.52%|+26.71%|+16.29%|
|long|512|−0.09%|+0.07%|−6.81%|+18.47%|+13.40%|
|long|1024|−0.77%|−0.52%|−17.95%|+9.45%|−4.07%|

For reference, the preceding (unfair/relaxed) GPU run reported 37.36% and
38.90% expert reductions at budgets 128/256.  In the corrected fair replay,
the routing-only component is at most 0.33 percentage points at those
budgets—roughly 0--1% of the prior absolute reduction.  The prior gain was
therefore dominated by the relaxed endpoint policy and its fewer/larger
invocations, with additional first-call timing bias in the preliminary run.

## 6. Fairness, correctness, and coverage

* Balanced and Same-count have exactly the Fixed invocation count.
* Same-count and Strict never exceed the hard budget; Relaxed is the only
  strategy allowed to exceed it.
* All routes have identical total assignment counts and contiguous token
  partitions.
* `correctness_all=true` and `route_identity_all=true` across all four EP
  rank files and all 880 observations/rank in the corrected run.
* Short route layers are route-ID slices replayed through the validated layer-24
  captured operator.  This is a grouped-MoE GPU replay, not a live 48-layer
  serving run; layer-specific activation/weight effects are out of scope for
  this decomposition and are explicitly not inferred.
* Long routes are included in GPU replay at layer 24 and all 48 layers in
  offline shape analysis.  They do not rescue the routing-only effect.

## 7. Figures and artifacts

Figures from the corrected replay:

* [plot1_gpu_decomposition.png](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/figures/plot1_gpu_decomposition.png)
* [plot2_chunk_size_fairness.png](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/figures/plot2_chunk_size_fairness.png)
* [plot3_component_breakdown.png](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/figures/plot3_component_breakdown.png)

Key artifacts:

* [offline_per_request_layer.csv](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/offline_per_request_layer.csv)
* [offline_decomposition.csv](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/offline_decomposition.csv)
* [strategy_cuts.json](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/strategy_cuts.json)
* [GPU strategy summary](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/gpu_analysis/strategy_summary.csv)
* [GPU decomposition](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/gpu_analysis/gpu_decomposition.csv)
* [GPU rank observations](../deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/gpu_analysis/per_observation.csv)

Primary result directory:
`poc_flashvep/deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/`

The exact offline command was:

```bash
python poc_flashvep/fair_chunk_oracle_decomposition/oracle.py
```

The corrected GPU command was:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 WARMUPS=5 ITERATIONS=20 FAIR_PREWARM=1 \
  poc_flashvep/fair_chunk_oracle_decomposition/run_gpu.sh \
  poc_flashvep/deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900
FAIR_ANALYSIS_RESULT=/home/esjung/MLLM-EP-github/poc_flashvep/deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900 \
  python poc_flashvep/fair_chunk_oracle_decomposition/analyze.py
```

## 8. Gate and interpretation

`FINAL STATUS: NO-GO` for a routing-only fair chunk oracle.  The preregistered
primary gate requires a median `Balanced → Same-count Oracle` GPU reduction of
at least 10% at both 128/256, positive signal at long 512/1024, and broad
request/layer consistency.  The observed reductions are below 1% and include
both signs.

Answers to the decomposition questions:

1. **Routing-only share of the earlier 37--40%:** effectively zero in the fair
   GPU comparison (maximum +0.33 percentage points at the primary budgets).
2. **Tiny-tail/size balancing:** not a robust source of GPU gain after
   prewarming and randomized measurement; its apparent first-run gain was
   measurement-order bias in the discarded run.
3. **Relaxation inflation:** yes.  Relaxed endpoints obtain the only sizeable
   reductions, principally through fewer/larger chunks and relaxed shape
   constraints.
4. **Strict opportunity:** no meaningful opportunity remains in this bounded
   exact-route replay; Strict is frequently slower because it needs more
   invocations.
5. **Scale robustness:** no routing-only GPU effect at 128/256/512/1024 in
   either short or long routes.
6. **Next-stage justification:** do not proceed to Real Serving Trace or a
   scheduler based on this route-aware boundary claim.  A serving trace would
   be justified only by a separately motivated chunk-size policy, not by the
   current routing-only evidence.

The result closes the fair route-aware chunk-boundary direction for this
workload and kernel protocol; it does not claim that all chunked-prefill
policies are unhelpful.

## 9. Final metadata

* `FINAL STATUS: NO-GO`
* branch: `flashvep/fair-chunk-oracle-decomposition`
* GPU mapping: physical 1,2,3,4 only
* source code: `poc_flashvep/fair_chunk_oracle_decomposition/`
* report: `poc_flashvep/reports/fair_chunk_oracle_decomposition.md`
* result: `poc_flashvep/deepep_revalidation/results/fair_chunk_oracle_decomposition_20260831_235900/`
