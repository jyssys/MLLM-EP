# Independent decision gates

The previous `CONDITIONAL_FIXED_QUANTUM_ONLY` verdict remains unchanged.
This study makes three separate decisions.
The final analysis reviewed the completed diagnostics before release.

| Decision | Assessment | Evidence and limit |
| --- | --- | --- |
| Overall study | CONDITIONAL | Repeated fixed-policy trade-offs exist. Quality equivalence and qualified serving capacity remain unproved |
| CHARACTERIZATION | CONDITIONAL | Full fixed grid and repeated shortlist support the distributional mechanism. Open-loop samples remain below 1,000 requests per arm/rate |
| ADAPTIVE_JOINT_HEADROOM | CONDITIONAL | Some fixed settings favor different objectives. No live Joint, adaptive C-only, or adaptive G-only comparison establishes additional online benefit |
| MULTIGPU_PORTABILITY | NOT_RUN | Four independent experiment workers are not a shared serving cluster |

## Phase gates

| Gate | Status | Reason |
| --- | --- | --- |
| G0: historical evidence audit | PASS | Eleven files and 375 archives matched. Actual token timestamps support request-SLO recomputation |
| G1: calibrated clocks and adequate tail samples | PARTIAL | Scheduled-arrival and token clocks are explicit. Request counts do not qualify request-tail P99 or production capacity |
| G2: fixed C/G characterization | PASS_WITH_LIMITS | The repeated shortlist separates rare severe stalls from frequent moderate stalls. Staged G4 isolates part of the execution-path effect |
| G3: serving-rate sensitivity | EXPLORATORY | Common arrivals, all-offered denominators, queue screens, and paired intervals are available. Finite windows cannot establish sustainable capacity |
| G4: live adaptive joint superiority | NOT_RUN | Required strong adaptive controls and quality evidence are absent |
| G5: shared multi-GPU serving portability | NOT_RUN | This task does not implement another serving architecture |

## Reasons not to implement Adaptive Joint now

1. Stock Mixed remains a strong decode-tail control.
2. Split C1024 and Mixed C1024 have fixed-grid data but no open-loop rate sweep.
3. A post-hoc best-fixed-policy panel does not demonstrate an online selection gain.
4. Batched output differences remain unresolved beyond the isolated exact probes.
5. GSM8K and MATH benchmark accuracy are NOT_RUN.
6. No tested arm/rate reaches the registered 1,000-request qualification threshold.
7. Shared-host and device effects limit absolute rate portability.

These limits do not prove that adaptive joint scheduling cannot help.
They prevent a quality-qualified GO claim from the current data.

## Smallest next experiment

First isolate the first divergent batched hidden state under matched request membership and graph state.
Keep BF16, model weights, R4, and output sampling unchanged.
Record the hidden state, logits, KV versions, stream events, and first divergent token.
Separate a numerical near-tie from a stale-state or synchronization error.

Only after that check should a serving comparison expand the strongest missing fixed controls.
Use Split C1024, Mixed C1024, Atomic C1024, and Grouped C1024 on common arrivals.
Select the SLO and rate bracket before reading those outcomes.
Increase complete-response samples near a plausible boundary.
Compare adaptive C-only before implementing Adaptive Joint.

The native-vLLM Ouro Loop PP program remains paused.
