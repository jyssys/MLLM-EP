# D-side encoder sharing and native Decode interference

The pinned D worker ran the original Qwen2.5-VL vision tower and returned exact LM-ready embeddings through CUDA IPC. The balanced six-image case reduced the encoder critical path by 88.38 ms in isolation. The next question was whether the same worker could accept that type of work during normal Decode or immediately before a native handoff. [Encoder and window probes](mllm_epd_idle_candidate_physical_screening.md)

## Controlled native handoff

E rendered and encoded an original natural Very-long input. P generated the full KV and published the actual NIXL handoff. D then received the same prompt and KV with a Fixed128 output contract. The table measures D HTTP submission to first SSE token and subsequent SSE token intervals. It excludes the completed E and P work. Each condition completed eight trials. Small and large stand for original fixtures with 1,620 and 19,200 processed vision patches. Those fixtures were auxiliary work; the Fixed128 request kept its original prompt and KV. [Raw intervention archive](../mllm_epd_modality_aware_idle_execution_discovery/evidence/vision-interference.manifest.json)

| Condition | D submit→first median | TPOT p95 | ITL p99 | ITL max | Absolute Decode passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Decode only | 97.09 ms | 4.848 ms | 5.450 ms | 6.99 ms | 8/8 |
| Small vision launched 3 ms before D submit | 150.80 ms | 4.853 ms | 5.380 ms | 5.85 ms | 8/8 |
| Large vision launched 3 ms before D submit | 179.23 ms | 4.914 ms | 5.594 ms | 8.36 ms | 8/8 |
| Small vision after first Decode token | 94.34 ms | 5.309 ms | 5.698 ms | 66.20 ms | 0/8 |
| Large vision after first Decode token | 95.83 ms | 5.776 ms | 6.107 ms | 123.26 ms | 0/8 |

The before-submit arms kept TPOT and ITL within the absolute guard and increased median D submit-to-first-token time by 53.71 and 82.14 ms. Native has a slow first-token outlier and its D-only p95 is 194.75 ms, so the small sample supports a median delay, with an uncertain tail effect at that boundary. In the active arms, Vision RPC median duration was 65.65 and 128.29 ms. Both arms violated the per-request requirement that every observed ITL stay at or below 34 ms. Their TPOT p95 grew 9.51% and 19.15%. Native output token IDs matched all 128 tokens in 8/8 paired trials for every intervention arm. These timings show a request-level collision with the existing Worker RPC path. The measurement covers stream lifetime and the actual work, with no timestamp proof of overlapping GPU kernels. [Interference figure](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/idle_physical_decode_interference.svg) · [Physical summary](../mllm_epd_modality_aware_idle_execution_discovery/results/vision_physical_summary.json)

## Critical-path decision

On the main offered load, 40/40 and 74/74 central E-ready events of the fresh N64 and N128 runs already had older P HTTP work outstanding. A fixed-observed-order replay advances every central E release by the full measured 88.38 ms without charging IPC or D interference. It gives no earlier P handoff in either main cohort. All of those are modelled handoffs, with effective service per original gap and preserved P completion order. At lower load the model advances 22/40 P handoffs, averaging 48.61 ms across all 40. The E-only benefit is therefore conditional on the pipeline state. Near saturation it lacks a measured mechanism for reducing P work or protecting D during the next handoff. [Native timeline and analytical bound](mllm_epd_encoder_critical_path_bound.md) · [Encoder and Prefill critical-path figure](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/idle_encoder_prefill_critical_path.svg)

Full-request co-encode and Prefill can remove P work as well as the E→P embedding hop. The independently observed whole D-local LM path alone took roughly 1.15 s, above the 0.86 s maximum upcoming-P-ready interval in the fresh main proxy replay with existing P work. Breaking that into valid prefixes requires carrying the original causal KV and comparing against adaptive token chunks. The historical generic E→D control is the measured end-to-end reference; no joint co-encode implementation was promoted. [Idle window scoring](mllm_epd_idle_motivation_qualification.md) · [Generic online controls](mllm_epd_decode_protected_online_serving.md)

The whole-image vision mechanism has a `CONDITIONAL_PHYSICAL_SIGNAL`. The active-Decode RPC mode is `NO_GO_DECODE_INTERFERENCE`. The screened mechanism on this saturated input was not promoted as an online modality candidate. Complete image parallelism has established prior art. An isolation-safe scheduler and a demonstrated effect on the P critical path would be needed to change the decision. [Prior-work comparisons](mllm_epd_idle_execution_prior_work.md)
