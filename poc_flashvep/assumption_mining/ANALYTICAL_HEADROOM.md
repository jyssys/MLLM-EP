# Analytical counterfactual headroom gate

No new GPU run is justified unless an assumption has a plausible repeated
request-level mass.  Estimates below use existing live traces and reports;
they are upper bounds, not achieved speedups.

| Candidate family | Most optimistic available bound | Gate result | Reason |
|---|---:|---|---|
| outstanding DeepEP tail / scoped wait | direct request tail excess 1.09%; old p25 projection 23.83% was not a direct join | DROP | wait-aware repeated policy had no robust E2E gain and throughput cost |
| communication-SM or backend config | HT lower envelope 0.11% median, 2.10% max (decode 0.25%) | DROP | below 3% even before implementation cost |
| metadata materialization | 0.1137 ms median per call; 48-layer nominal <5.5 ms | DROP | <1% of 0.3--1.5 s request medians; rare max is not mass |
| DP dummy/rendezvous/partition | -0.26% dummy, -0.03% NCCL switch, -1.83% async-off | DROP | measured controls are null or negative |
| phase/composition/turnover | at most 2.5% request-level in surviving controls | DROP | wave deltas do not transfer to request critical path |
| fanout/incidence geometry | Model2→3 RMSE +0.001% | DROP | no incremental information after load features |
| vision/image composition | 0.3--2% after controls; previous overlap was negative | DROP | front-end/token-volume effect, not structural EP mass |
| partial/speculative execution | prior oracle about 11.4% at best; verification and real pipeline gates fail | WEAK/DROP | below >=15% promote threshold and crowded prior art |
| arbitrary new ownership/materialization contracts | no measured mass | UNKNOWN | requires a new observable; not a reason to occupy GPUs |

The conservative decision is to stop before GPU: no candidate simultaneously
has measured or defensible >=20% analytical direct E2E headroom, a non-trivial
counterfactual, and no close prior-art collision.  The UNKNOWN rows are
explicitly retained as future work, not silently called negative.
