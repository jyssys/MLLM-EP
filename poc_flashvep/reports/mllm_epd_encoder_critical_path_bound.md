# Encoder-only critical-path bound

An optimistic 88.38 ms encoder release advance gives earlier P handoffs for 0/40 in `fresh_native_n64`, 0/74 in `fresh_native_n128`, 22/40 in `fresh_native_lower_n64` in the propagated fixed-order replay.

The physical motivation is the measured balanced whole-image probe with some independent images executed on D. The replay applies its entire latency reduction to every central E-ready request while charging zero extra movement, scheduling, or D contention. The primary cases preserve their original images, arrival times and no-assist native route.

## Observed host timeline · TRACE_ANALYSIS

These clean traces contain the actual encoder acknowledgment, P HTTP submission, and P-ready acknowledgment. Per-request P engine dispatch requires additional attribution. 0/256 P HTTP responses carry metrics. The last sampled P waiting count has a maximum allowed age of 750 ms. P waiting counts and outstanding HTTP enclosures describe different queue boundaries.

| Cohort | Central E-ready events | Earlier P HTTP outstanding at E-ready, positive / median | P waiting sample positive / coverage | P waiting median | P completion/submission inversions |
| --- | --- | --- | --- | --- | --- |
| `fresh_native_n64` | 40 | 40/40 / 8.00 | 40/40 | 5.00 | 13 |
| `fresh_native_n128` | 74 | 74/74 / 14.00 | 68/74 | 8.00 | 33 |
| `fresh_native_lower_n64` | 40 | 35/40 / 2.00 | 17/40 | 0.00 | 0 |

## Fixed observed-order replay · ANALYTICAL_BOUND

Order all requests by measured P-ready time. For each request use effective service `observed P-ready − max(P HTTP submit, previous observed P-ready)`. This reconstructs the original handoff series exactly when no release changes. It assigns each gap to a serial predecessor and can absorb batching and other stage costs into that gap. The effective service is a model quantity.

Advance each central P release by the full probe gain, capped at its recorded E enclosure. The isolated counterfactual keeps every preceding native completion fixed. The propagated counterfactual applies all the central release advances, and each earlier modeled handoff can move the next service gate. Prior work outside the intervention keeps its native release. Output times refer to modeled P-ready. D-side scheduling effects remain open.

| Cohort | Isolated improved | Isolated mean / max ms | Propagated improved | Propagated mean / max ms |
| --- | --- | --- | --- | --- |
| `fresh_native_n64` | 0/40 | 0.00/0.00 | 0/40 | 0.00/0.00 |
| `fresh_native_n128` | 0/74 | 0.00/0.00 | 0/74 | 0.00/0.00 |
| `fresh_native_lower_n64` | 5/40 | 11.05/88.38 | 22/40 | 48.61/88.38 |

The main cohorts contain overtakes in the observed P-ready order. Therefore this replay preserves observed completion order rather than assuming submission FIFO. The lower-load case happens to preserve submission order. A change that rearranges P work or changes P batching could affect individual requests. Establishing that effect requires P dispatch/completion attribution and an actual matched intervention.

[Raw-derived summary](../mllm_epd_modality_aware_idle_execution_discovery/results/encoder_critical_path/encoder_critical_path.json) · [Per-request reduction](../mllm_epd_modality_aware_idle_execution_discovery/results/encoder_critical_path/encoder_critical_path_rows.csv)
