# Anomalies and generated children

## ANOMALY-01 — concurrency-dependent phase mix

Fixed text c8→c16 increased measured prefill `T_MoE` p50 from 1.168 to
1.399 ms (+19.8%); dispatch share rose 30.4→28.4% only slightly while event
wait share rose 3.3→14.6%. Request E2E p50 rose 656→773 ms (+17.8%). This is
not yet causal because batch size, scheduler state, and request count change
together. Children: C1a randomized fixed-text c2/c8/c16; D2a stage-isolation.

## ANOMALY-02 — vision changes expert share without large T_MoE shift

At c8, high-resolution vision has expert share 42.2% vs fixed text 31.0%, but
T_MoE p50 differs only 4.3% (1.218 vs 1.168 ms). The E2E gap (15.5%) includes
vision preprocessing/embedding and cannot be called an MoE method headroom.
Child: G1a matched-M/layer route control.

## ANOMALY-03 — dispatch-only extreme maxima

Fresh traces retain 1.2–2.6 s dispatch maxima while expert/combine maxima stay
near 3–31 ms. This reproduces the historical async-tail signature. It is
closed as a research direction because the direct request-level bound was
1.09% in the prior joined trace; it is retained as an engineering diagnostic.

## ANOMALY-04 — model residuals near zero explanatory power

Time-block linear models using M, expert distribution, rank loads, fanout have
held-out R² approximately 0 and fanout adds at most 0.001% RMSE reduction (and
often worsens error). Children must introduce a genuinely new variable rather
than reusing fanout/load.

## Child-generation rule

Each anomaly creates a bounded child only when its direct E2E mass is not
already killed. A candidate with a large micro-level spike but no request-level
mass remains diagnostic-only.

## ANOMALY-05 — residual tail is high-M / high-active-expert, not fanout

Across 90,240 fresh logical rows, the top 1% residual cluster is enriched by
about +181 scheduled tokens and +11 active experts relative to its matched
baseline, while rank CV and fanout move only modestly. This is consistent with
ordinary shape/kernel cost and fails the orthogonal-variable gate; it does not
revive the closed fragmentation or histogram directions.

## ANOMALY-06 — cross-modality shape-transition state

Fresh controls with the same Qwen3-VL worker and DeepEP HT show a persistent
state change at the vision/text boundary. Text fixed after text warmup has
T_MoE p50 1.161 ms and event-wait p50 0.018 ms; high-resolution vision fixed
after matching vision warmup has 1.204 ms/0.028 ms. Text warmup followed by
high-resolution vision gives 2.395 ms/0.334 ms, and alternating text and
vision gives about 2.12 ms/0.49 ms. Alternating two text shapes remains
1.21–1.25 ms/0.02–0.04 ms. The trigger is therefore tied to the multimodal
encoder/runtime boundary rather than shape count alone. Matching the warmup
shape removes the effect.

**Candidate C1c:** multimodal shape-transition state sensitivity.

**Mechanism hypothesis:** switching between the vision encoder and text path
leaves communication/workspace/stream state that adds DeepEP wait/dispatch
overhead until the matching shape is re-established. GPU clock ramp
(345→1980 MHz in telemetry) is a recorded cofactor.

**Disposition:** high-headroom diagnostic but `TRIVIAL_ENGINEERING` under the
headroom/novelty gate; static shape-aware warmup/bucketing is an obvious fix.
Vision-only 2↔3 alternation (T_MoE p50 1.16ms) and a 32-wave text
shape-matched control (12,288 logical rows, p50 1.175ms) further support this
disposition. A fresh telemetry-tagged text→vision replication subsequently
returned to 1.20ms T_MoE / 0.04ms wait (64 requests, clocks 345→1980MHz), so
the large transition effect is not statistically robust across independent
runs and is likely coupled to external runtime/DVFS/cache state. No production
method is implemented.
