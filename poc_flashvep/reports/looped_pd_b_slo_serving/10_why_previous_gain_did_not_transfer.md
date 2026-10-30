# Why the earlier B SLO signal did not transfer

## 1. Final answer and evidence scope

B is not accepted as a lossless serving optimization in this campaign.
The original runtime fails paired output qualification in both models.
The canonical-runtime repair improves repeatability but does not remove every mismatch.
Its independent Ouro serving confirmation fails399/400 after an earlier400/400 pass.
Huginn fails the repaired200-request qualification at199/200, with198/200 in the control repeat.

The earlier B result remains a valid, conditional observation in its stated workload and SLO cell.
It was not evidence of a universal latency reduction or sustainable serving-capacity improvement.
The earlier final report already states these limits.
This report does not rewrite or invalidate that frozen evidence.

The frozen prior commit is `b82eabd20a4ec168a5ad9a119c8401f97a565e78`.
Read its [Track B decision](https://github.com/jyssys/MLLM-EP/blob/b82eabd20a4ec168a5ad9a119c8401f97a565e78/poc_flashvep/reports/looped_pd_abc/07B_final_decision.md).
Read its [final summary](https://github.com/jyssys/MLLM-EP/blob/b82eabd20a4ec168a5ad9a119c8401f97a565e78/poc_flashvep/reports/looped_pd_abc/09_final_summary.md).

No new GPU experiment starts during this closeout.
The new sensitivity table rescores existing emitted-token timestamps only.
`TRACE_RESCORING` is not a new online serving run or a validated counterfactual scheduler.
`STRUCTURAL_ACCOUNTING` is not measured E2E speedup.
All unmatched or unexecuted cells remain unaccepted.

## 2. The studies answer different questions

| Field | Earlier accepted B comparison | Current B follow-up |
|---|---|---|
| Models | Ouro R4 and Huginn R32 | Same model revisions and fixed depths |
| Precision | BF16 | BF16 |
| Main workload | Fixed input512, concurrency4, cap32 | Real ShareGPT length mixture, Poisson arrivals, cap128 |
| Prompt population |24 paired synthetic-text prompts per start |200 distinct real-text prompts |
| Replication |3 starts,72 evaluations of24 prompts | Original200-request ladder and independent confirmations |
| Additional repaired serving | Not applicable |2 starts of400 arrivals, each with two full cycles of200 prompts |
| Prominent historical SLO | TTFT<=5000ms and every observed ITL<=150ms, plus E2E<=30000ms | Baseline-calibrated TTFT and request-average TPOT |
| Current Ouro primary SLO | Not the historical selected cell | TTFT<=857ms and mean TPOT<=30ms |
| Current Huginn primary SLO | Not the historical selected cell | TTFT<=1454ms and mean TPOT<=119ms |
| Goodput denominator | Sum of finite cohort spans | Frozen open-loop arrival-window seconds, with drain reported separately |
| Primary acceptance | Conditional bounded-workload signal | Largest tested stable rate with>=90% joint attainment |
| Output gate |72/72 in the named old comparison | Original185/200 and193/200 failures, followed by separate repair tests |
| Runtime after repair | Not applicable | Same canonicalization applied to B and its new-epoch controls |

Both studies preserve EOS and disclose output caps.
Neither capped workload establishes full-answer completion quality.
Changing a request-maximum ITL condition to a request-average TPOT condition changes what counts as success.
It is not valid to compare the two percentages as the same endpoint.

## 3. How large was the earlier signal?

The historical Huginn P512/C4/cap32 cell uses TTFT5000ms and maximum observed ITL150ms.
Its numbers below come from the frozen `paired_SLO_comparisons.csv`.
The listed95% intervals are conditional, not selection-adjusted capacity intervals.

| Comparator | Qualified comparator/B | Comparator goodput req/s | B goodput req/s | B goodput ratio |
|---|---:|---:|---:|---:|
| DP |18/72 versus70/72|0.314211|1.047915|3.335×|
| DP_Mixed |9/72 versus70/72|0.154278|1.047915|6.792×|
| PD |53/72 versus70/72|0.688092|1.047915|1.523×|
| PD_G1 |56/72 versus70/72|0.799664|1.047915|1.310×|

PD_G1 is the strongest listed goodput control in this specific cell.
The1.310× matched-control ratio has the prior95% CI[1.125,1.580].
The much larger6.792× ratio compares B with a baseline that meets the maximum-gap rule on only9/72 requests.
It does not mean the GPU processes requests6.792 times faster.

The measured-count decomposition is:

`goodput ratio = qualified-count ratio × cohort-span ratio`.

For Huginn B versus PD_G1, `1.310443 = 1.250000 × 1.048355`.
The first factor is70/56. The second factor is the ratio of summed observed cohort spans.
Most of the apparent31% goodput increase is associated with crossing a binary SLO threshold.
It is not a31% reduction in raw processing time.
This is `STRUCTURAL_ACCOUNTING` of measured counts and spans, not a separate speedup experiment.

The old E2E evidence also limits the interpretation:

| Model and regime | Strong E2E control | Control/B p95 E2E ms | B speedup [conditional95% CI] |
|---|---|---:|---|
| Ouro P512/C4/cap32 | DP_Mixed |1070.138 /981.602|1.090× [1.012,1.153]|
| Huginn P512/C4/cap32 | DP |3295.142 /4065.456|0.811× [0.761,0.896]|
| Ouro P512/C4/cap128 | DP_Mixed |2842.743 /2898.505|0.981× [0.970,1.002]|
| Huginn P512/C4/cap128 | DP_Mixed |11883.670 /13319.751|0.892× [0.881,0.987]|

Thus, even the earlier study does not show general E2E superiority over colocated serving.
Ouro's earlier goodput versus DP_Mixed is1.034× [0.975,1.143], not a demonstrated large gain.
The prior report also contains losing C1 and longer-context regimes.

## 4. Direct CPU rescoring isolates the SLO predicate

`historical_vs_current_slo_rescoring.csv` contains the full rescore.
`closing_analysis.py` reproduces it without CUDA execution.
The historical inputs, token outputs, timestamps, and cohort spans remain unchanged.

### 4.1 Historical Huginn traces, different predicates

| Method | Historical max-ITL150ms rule | Current TTFT1454ms + mean-TPOT119ms rule | Goodput under current rule req/s |
|---|---:|---:|---:|
| DP |18/72|72/72|1.256845|
| DP_Mixed |9/72|72/72|1.234220|
| PD_G1 |56/72|72/72|1.028140|
| B |70/72|72/72|1.077855|

Under the current predicate, the historical B/PD_G1 ratio becomes1.048×, not1.310×.
Against historical DP_Mixed under that predicate, B gives0.873×.
Against historical DP, B gives0.858×.
These are `TRACE_RESCORING` ratios with original finite-cohort denominators.
They do not establish current open-loop capacity.

The maximum-gap rule penalizes even one interruption during a request.
The average-TPOT rule averages observed generation duration across its output intervals.
A policy can improve the first condition without improving average service time.
The strongest baseline can therefore change when the objective changes.

### 4.2 Current original-runtime Huginn W1, both predicates

These runs use200 real-text requests, rate0.2req/s, seed2026103001, and output cap128.
B fails193/200 output parity. The counts below are descriptive and not accepted lossless performance.

| Method | Historical max-ITL rule, including TTFT5s/E2E30s | Current primary rule |
|---|---:|---:|
| DP |87/200|193/200|
| DP_Mixed |93/200|187/200|
| PD_G1 |177/200|169/200|
| B |166/200|162/200|

Even the old predicate does not restore B's lead on the new W1 trace.
B/PD_G1 qualifying-count ratio is166/177, or0.938×, with a shared arrival-window denominator.
Therefore, the SLO change alone does not explain the complete reversal.
Prompt content, length distribution, arrival pattern, active batch composition, and generation duration also changed.
This campaign does not independently vary all those factors.
It cannot assign a causal percentage to each factor without further matched experiments.

For Ouro's new original-runtime W1, the old predicate yields190/200 for both PD_G1 and B.
The current predicate yields190/200 and191/200 respectively.
That difference is small, and B still fails185/200 output parity.

## 5. B does not eliminate the dominant observed phase

The following numbers use actual transport-counter deltas and host notification timings.
They are not exact GPU DMA durations.
The source keeps the same BF16 payload. B changes its transfer frontier and fragmentation.

| Original W1 pair | PD_G1 payload bytes | B payload bytes | PD_G1/B fragments | PD_G1/B observed cumulative transfer seconds |
|---|---:|---:|---:|---:|
| Ouro, rate1,200 requests |131400237056|131400237056|2058 /3242|49.138 /62.796|
| Huginn, rate0.2,200 requests |460435430400|460435430400|7183 /11114|54.510 /105.413|

Cumulative transfer observations overlap. They must not be added to E2E.
More fragments can increase descriptor, event, polling, and synchronization work.
That is a plausible mechanism, not a measured exclusive explanation of the entire slowdown.

### 5.1 Actual host-phase shares

The denominator is summed valid request E2E wall time, not cohort makespan or GPU utilization.
Each row below uses all200 or400 requests in its named pair.
Phases include their internal scheduling and synchronization costs.

| Observed phase, % | Original Ouro G1/B | Original Huginn G1/B | Canonical Ouro G1/B, seed6001 |
|---|---:|---:|---:|
| Coordinator queue |0.0050 /0.0053|9.9914 /10.5252|0.0039 /0.0037|
| Reservation/control |0.2857 /0.2865|0.0436 /0.0439|0.3680 /0.3946|
| P prefill, including backpressure |9.5180 /7.9439|3.8288 /3.5228|10.7847 /11.4444|
| Exposed handoff |0.5509 /0.2827|0.0900 /0.0252|0.6044 /0.6060|
| D resume through first token |0.8574 /1.7244|0.0183 /0.0182|0.5483 /0.7662|
| Remaining decode |88.7521 /89.7286|86.0217 /85.8580|87.6633 /86.7603|
| Harness residual |0.0309 /0.0285|0.0062 /0.0068|0.0274 /0.0249|

Small rounding differences can affect the displayed sum. The CSV retains full precision.
The original B columns remain output-confounded.
The canonical seed6001 pair is400/400 output-exact but lacks independent passing confirmation.

Original Huginn exposed handoff falls from11.914ms to3.386ms on average.
This is a real host-observed interval reduction in that pair.
However, that interval occupies only0.0900% of the PD_G1 summed request wall time.
Removing only that interval cannot establish a large E2E improvement.
This bound does not cover indirect queue, prefill-backpressure, or batch-composition effects.

The canonical Ouro pair does not show a handoff improvement.
Its mean is21.029ms for G1 and21.865ms for B. P95 is66.517ms versus87.707ms.
Its payload is262800474112 bytes in both arms, with4116 versus6486 fragments.

GPU-only Attention, FFN, communication, and idle critical-path percentages are `NOT_MEASURED`.
Native heavy profiler data and fine-grained transfer event traces are also `NOT_MEASURED` at closeout.
The available host phases must not be relabeled as GPU computation or dispatch percentages.
`e2e_component_percent.csv` preserves this distinction.

## 6. What the repair fixes, and what it does not

The small Ouro reproduction finds identical prompt KV but different historical decode KV before the diverging token.
Its original comparison differs in551940 BF16 KV elements, despite identical generated history through index124.
Canonical linear and masked fixed-capacity graph execution reduces that captured difference to0.
The small reproduction then matches8/8 outputs.

The candidate also passes the original Ouro200-request comparison and identical-control repeat at200/200.
Transferred prompt-KV/final-hidden checks pass8/8 with reversed fragments.
Observed recurrent-row counters match full-R4 structural accounting in all three200-request runs.

The improvement does not generalize completely:

| Gate | Original | Canonical candidate |
|---|---:|---:|
| Ouro first200-request B/control |185/200|200/200|
| Ouro independent400-arrival serving, seed6001 | Not the same trace |400/400|
| Ouro independent400-arrival serving, seed7001 | Not the same trace |399/400|
| Huginn first200-request B/control |193/200|199/200|
| Huginn identical-control repeat at200 | Separate earlier prefix evidence |198/200|

The residual Ouro case is sample233, with first divergence at output index4.
Its two output lengths remain128. Equal lengths do not establish identical computation.
The first residual operator and whether it shares the initial case's cause remain unresolved.
Do not infer that every discrepancy is harmless BF16 noise or that all KV/state bugs are excluded.

Canonicalization also changes the control's output relative to the original run:142/200 exact Ouro sequences.
This is not an accuracy score. The original engine also has self-repeat sensitivity.
The fresh canonical Native/HF probe matches7/8 Ouro sequences, not8/8.
The original Huginn Native/HF probe remains6/8.
Full downstream answer accuracy is `NOT_MEASURED`.
No original-model-lossless claim is accepted.

## 7. Actual repaired serving result

These are isolated two-GPU natural-generation runs at1req/s, using identical canonicalization in all three arms.
Seed6001 uses GPUs0,1. Seed7001 uses GPUs2,3 with crossed arm order.
Each start has400 arrivals and200 distinct source prompts repeated twice without prefix caching.

| Seed | DP_Mixed joint attainment | PD_G1 joint attainment | B joint attainment | B/G1 output parity |
|---|---:|---:|---:|---:|
|2026106001|175/400,43.75%|349/400,87.25%|321/400,80.25%|400/400|
|2026107001|213/400,53.25%|298/400,74.50%|273/400,68.25%|399/400|

All three methods fall below90% in both starts.
The exact first pair gives B/G1 SLO-goodput0.920×.
It has only one fully exact independent start, so its between-start95% CI is `NOT_MEASURED`.
The second point is output-confounded and cannot validate an equivalent-work speedup.

In the exact first pair, G1/B p95 TTFT is876.059/1082.067ms.
P95 TPOT is30.541/30.872ms. P95 E2E is4419.234/4623.385ms.
B's p95 E2E speedup is approximately0.956× at this point, not a gain.
Canonicalization can add GPU work and change batching behavior.
Its costs apply to every new-epoch arm. These results must not pool with original-runtime measurements.

## 8. Final decision and next research step

The method-level outcome is `NO_GO_CORRECTNESS` for lossless B acceptance in the measured two-GPU studies.
The exact repaired Ouro point is also negative performance evidence at its tested rate.
This does not prove that no workload can benefit from frontier streaming.
It does show that the previous selected SLO result does not establish the current goal.

There is no accepted B capacity or same-budget system speedup.
The original baseline rate grid gives bounded raw passing points, not a universal steady-state maximum.
Original Ouro G1 passes a tested1req/s bound, versus0.5req/s for DP_Mixed.
Original Huginn G1 passes0.1req/s, while DP_Mixed passes0.2req/s.
Their original-model fidelity and inter-policy output limitations remain explicit.

Four/eight-GPU B is `NOT_MEASURED`, not a measured hardware failure.
Baseline scale probes are present but have one start and100 requests per cell.
Generic immutable-layer streaming is not independently implemented. Loop-exclusive novelty remains `NOT_ESTABLISHED`.

The next useful experiment is not another broad B capacity sweep.
First reproduce the residual Ouro sample233 and the residual Huginn requests with exact operand/KV evidence.
Use batch-permutation and geometry-invariance tests across prefill, prelude, core, coda, and sampling boundaries.
Keep the original checkpoint, BF16 precision, fixed depth, and strict output gate.
Only a qualified, matched backend should support another lossless serving claim.
If that succeeds, use a matched predicate/workload crossover to isolate SLO and arrival-pattern contributions.

## 9. Reproduction and publication

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/esjung/.venvs/looped-pd-abc/bin/python \
  -m poc_flashvep.looped_pd_b_slo_serving.closing_analysis
```

Raw evidence is under `raw/`. Per-run contracts and commands remain unchanged.
`04_latency_percentiles.csv`, `06_transfer_and_critical_path.csv`, and `e2e_component_percent.csv` retain the detailed measurements.
The first local commit failed GitHub's100MB file limit for the redundant extended dataset.
Its local original and unpublished commit remain preserved.
The publication replaces only the Git representation with a19,381,397-byte gzip file.
Decompression reproduces all126,308,939 original bytes and their SHA-256 exactly.
Read `configs/dataset_storage.json` and `reproduce.md`.
No prompt, arrival, model, or measured result changes for this packaging operation.

The first automatic closeout occurred13.954hours after setup. Its first push then failed.
The user resumed report/publication work at13:04KST after a capacity interruption.
This administrative continuation exceeds the original15-hour total allowance. That total allowance was not fully met.
No GPU inference experiment is restarted during the continuation.
The initial closeout timestamp remains preserved. It must not be used as the later publication completion time.
