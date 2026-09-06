# Live anomalies

## ANOMALY_01 — H09 common latency-regime transition

During the live randomized H09 chunk-residue control, both variants initially
had request medians near 680–690 ms. Around pair 300 the common level dropped to
approximately 450–480 ms while A and B remained close. This is a synchronized
state transition, not a positive A/B treatment effect. Candidate causes are
warm kernel/workspace state, scheduler/KV allocator state, or a GPU clock
transition; first-case ordering is not sufficient to distinguish them.

Follow-up: after the main campaign, repeat H09 with a fresh worker and record
per-step host/CUDA stage medians in 30-second segments, then repeat with a
no-hook worker if the transition recurs.

## ANOMALY_02 — H14 common decode-state expansion

Near the end of H14, both variants expanded from about 9.0 s to 13.6 s in
adjacent waves. Because A and B moved together, this is a common serving-state
transition rather than a context-length treatment effect. It will be compared
with H09's common drop using segment-level step/host data.

## ANOMALY_03 — v7 common-state persistence across turnover and mixed phase

The fresh v7 H46 turnover block entered a shared elevated regime (A roughly
5 ms, B roughly 9 ms after an earlier 3.4/6.5 ms level), and H48 began in the
same roughly 10 ms state. A/B request medians remained close despite the large
wave completion spread. This strengthens the interpretation that the earlier
H09/H14 transitions are worker/runtime state drift, not a hidden request-shape
or MoE intervention effect. It is retained as an observability question only;
no high-mass causal control was found in this sprint.
