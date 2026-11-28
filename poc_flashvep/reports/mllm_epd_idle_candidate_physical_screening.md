# D-side physical screening on H200

We kept the Qwen2.5-VL-7B BF16 vision model, FA3, original processor geometry, window order and language input width. GPU 4 was E and GPU 6 was D. Both worker processes exposed the original 1,353,100,288-byte vision parameter set. The tests ran inside these Native EPD processes on the same H200 group. Frozen inputs contain actual processed values from the primary natural dataset. The path benchmarks begin after the images have already been processed. They measure GPU placement, computation, copies, acknowledgement and joins within those stated boundaries. [Run specification](mllm_epd_modality_aware_idle_execution_poc_spec.md) · [Raw physical evidence](../mllm_epd_modality_aware_idle_execution_discovery/evidence/vision-physical-retry.manifest.json)

## A. Complete images

In seven fixtures, four repeated D outputs per fixture were bitwise equal to their E references. These 28 runs test the original LM-ready embedding tensor. The two-way case divided six images into two groups of three. The D group contributed 5,821 merged visual tokens and E received an actual 41,724,928 B CUDA IPC payload. The full group output after reassembly matched E-only bitwise in 4/4 repeats. E-only versus balanced E/D wall time moved from 274.46 to 186.08 ms, a 32.2% reduction. This is evidence that the original vision tower can use a short physical D slot and reduce an encoder batch's critical path. It leaves LM Prefill work unchanged. [Physical summary](../mllm_epd_modality_aware_idle_execution_discovery/results/vision_physical_summary.json)

| Fixture | Vision patches | D vision GPU median (ms) | D→E full round p95 (ms) | CUDA IPC returned (B) |
| --- | ---: | ---: | ---: | ---: |
| Image 0 | 1,620 | 70.38 | 238.78 | 2,903,040 |
| Image 1 | 1,620 | 70.15 | 145.09 | 2,903,040 |
| Image 2 | 2,464 | 70.58 | 101.60 | 4,415,488 |
| Image 3 | 2,464 | 70.24 | 151.00 | 4,415,488 |
| Image 4 | 19,200 | 113.72 | 209.27 | 34,406,400 |
| Image 5 | 20,160 | 117.89 | 255.75 | 36,126,720 |
| Group of six | 47,528 | 266.90 | 307.56 | 85,170,176 |

The round includes pinned-host upload and cast, original vision execution, the return copy into E-owned storage, an ordered reference check and explicit retirement ACK. Small-sample p95 values reflect host and GPU variation as well as the operation. Image and patch counts alone did not predict the observed round time. The historical independent E/P references were 1.716 and 1.221 req/s on this cohort. Faster E consequently has to demonstrate that its newly ready work avoids actual P waiting. The full image split is also an established encoder mechanism. [Physical time plot](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/idle_physical_image_work.svg) · [Prior-work controls](mllm_epd_idle_execution_prior_work.md)

## F. Offload independent windows before the first global vision block

`MEASURED_PHYSICAL` on 7 fixtures × 3 cuts × 4 repeats. E performed the stock patch projection and window ordering. E and D divided the original contiguous windows and executed blocks 0–6 with the same grid, lengths and RoPE. The return was consumed on E before block 7. E then executed the remaining layers, merger and inverse permutation. The peer path transferred a BF16 1,280-wide activation in each direction through CUDA IPC. The comparison includes the two copies, payload lifetime, join and acknowledgements. [Execution DAG](mllm_epd_execution_dag_and_resource_analysis.md)

All 21 cuts were slower than the paired native E path at their four-repeat medians. The best six-image point kept 35,624 rows on E and sent 11,904 to D. D's prefix kernel phase took 16.43 ms and the round transferred 60,948,480 B. The complete path was 290.95 ms versus 274.74 ms on E-only, a 5.9% regression. The phase-overlap estimate was 271.63 versus 269.62 ms. This estimate takes the maximum of the concurrent phases and omits control RPC overhead while retaining the measured copies. Even by that estimate, the best cut was 0.75% slower. The maximum observed relative L2 difference from the native embedding was 0.09559. Native E's unsplit frontier matched 84/84. E-only execution with the two split shapes and the physical E/D result each matched 28/84. All 84 E-only split and E/D trials produced the same recorded error summaries. The first observed divergence is reproducible on E without a peer link. [Physical and estimated placement comparison](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/idle_window_partition_transfer.svg)

**`NO_GO_COMMUNICATION` for this packed-activation placement.** Copy and join costs erase the independent compute. `NO_GO_CORRECTNESS` also applies to the shapes with numerical divergence. The path misses the physical timing gate even at bitwise-correct points.

The best measured path retained a 218.15 ms E tail after the join. A second `ANALYTICAL_BOUND` deletes both peer-copy phases and all control time from each repeat. It preserves the measured CUDA phases and the mandatory E tail. The best candidate then moves 269.62→245.84 ms, an 8.82% reduction at the 23,744-row cut. This estimate falls below the initial 10% goal for the tested frontier and execution shapes. [Budget comparison](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/idle_physical_work_budget.svg)

## Other screened paths

The candidate audit keeps the different mechanism, strongest control, legal dependency and fastest discriminator for each A–G path. The original worker path returns embeddings in scheduled groups and publishes them item by item after group computation. Its earlier positive transport-ready lead was 10/24 requests, with p95 85.7 ms. That lead leaves a potential smaller causal consumer boundary, with no demonstrated independently scheduled LM progress. Generic whole-request D-local Prefill historically took about 1.15 s to return its first token. That exceeds the observed P-backed windows on this workload. The full E/D local chain would also run Vision without an independently measured safe deadline. Prior slice-aware transport is the control for image/token boundaries. A new transport saving or latency win for C, D, E or G was not measured. [Candidate-by-candidate audit](mllm_epd_autonomous_candidate_discovery.md) · [Idle-window evidence](mllm_epd_idle_motivation_qualification.md)
