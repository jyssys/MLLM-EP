# DeepEP wait-aware tail PoC

## Decision

**FINAL STATUS: NO_GO (selective policy not validated; native readiness API unavailable)**

The causal mechanism from the preceding fixed-shape investigation is
reconfirmed: a same-device event-wait proxy is strongly enriched in dispatch
tails, while expert and combine remain small.  However, the installed DBO-off
runtime supplies no Python `previous_event` object to the HT path, and
DeepEP's C++ `EventHandle` exposes no nonblocking readiness query.  A
communication-stream drain is therefore only a coarse proxy intervention.  It
was not robust across independent runs, and neither the offline invocation-ID
oracle nor the simple previous-dispatch policy met the E2E gate.

## Environment and source path

* GPUs: physical 1,2,3,4 only (`CUDA_VISIBLE_DEVICES=1,2,3,4`); all four are
  H100 80GB and are fully NV18 connected.  Other GPUs were not touched.
* Model: Qwen3-VL-30B-A3B-Instruct snapshot
  `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c`.
* Runtime: local vLLM 0.20.0+cu129, BF16, V1, TP2/DP2/EP4, DeepEP
  `deepep_high_throughput`, Triton/unquantized experts, linear placement,
  eager, DBO off, prefix cache off.
* The exact source audit is in the timestamped result's
  `source_audit.md`.  `deepep_ht.py` calls `dbo_get_previous_event` before
  layout/dispatch, waits on returned dispatch events before experts, and uses
  a second previous event for combine.  With DBO disabled,
  `dbo_get_previous_event` returned `None` in all 773,760 observed rows.

## Measurements

The new local hook added same-device CUDA events around layout, dispatch,
expert, combine, and every stock `EventOverlap.current_stream_wait()` call. It
also records event presence, wait durations, communication-stream ID, and
diagnostic intervention metadata.  No cross-device timestamp subtraction was
performed.

There were 15 completed serving runs and 773,760 invocation records.  The
decode-heavy primary set contained 3 stock, 3 always-sync, 3 unconditional
comm-drain, 3 invocation-ID oracle, and 3 previous-dispatch-threshold simple
policy runs (run sizes are shown in `policy_run_summary.csv`).  Raw JSONL and
route traces remain in the timestamped result directory; large raw files are
not added to git.

### Event-readiness proxy

Across the three stock runs (160,704 decode rows), the closest observable
same-device wait proxy had p50 0.0198 ms, p99 0.8879 ms, and max 5.02 ms.
For a 1 ms proxy threshold:

* `P(dispatch > 10 ms | wait <= 1 ms) = 0.050%`;
* `P(dispatch > 10 ms | wait > 1 ms) = 10.441%`;
* relative risk ≈ **208×**.

Tail rows (`dispatch > 10 ms`) had wait-proxy median 1.359 ms versus 0.0198
ms for normal rows.  This is strong attribution to outstanding communication
state, but it is not a native previous-event readiness bit: the internal C++
dispatch wait is not directly observable by the Python API.

### Stage decomposition (pooled stock decode rows)

| stage | p50 (ms) | p99 (ms) | p99.9 (ms) | max (ms) |
|---|---:|---:|---:|---:|
| whole MoE | 1.016 | 3.661 | 15.188 | 2062.456 |
| layout | 0.037 | 0.074 | 0.138 | 0.257 |
| dispatch | 0.120 | 1.806 | 11.577 | 2059.775 |
| expert | 0.407 | 0.738 | 1.492 | 4.753 |
| combine | 0.042 | 0.089 | 0.154 | 0.334 |
| observed downstream event wait | 0.020 | 0.888 | 2.021 | 5.022 |

The giant tail is dispatch-dominant.  Because the internal notify/barrier
wait happens inside the DeepEP C++ dispatch call, the downstream wait proxy
underestimates the full dependency wait for the largest events.

### Policy medians across independent runs

| policy | MoE p50 (ms) | MoE p99 (ms) | dispatch >20 ms | throughput (tokens/s) | interpretation |
|---|---:|---:|---:|---:|---|
| STOCK (3) | 1.001 | 3.653 | 0.0439% | 1.554 | baseline |
| ALWAYS_SYNC (3) | 0.996 | 3.509 | 0.0868% | 1.408 | global diagnostic; no reliable tail advantage in this sample |
| ORACLE selective comm drain (3) | 1.687 | 3.769 | 0.0409% | 1.310 | p50/throughput cost; only ~7% extreme-tail reduction |
| ONLINE_SIMPLE, prev dispatch ≥1ms (3) | 0.989 | 3.311 | 0.0664% | 1.460 | p99 improved ~9%, but >20ms tails increased and throughput fell ~6% |
| unconditional comm-stream drain (3) | 1.046 | 5.719 | 0.0998% | 1.337 | unstable; one run had 5.96s dispatch max |

The prior root-cause diagnostic sync run (separate artifact) did eliminate
`>20 ms` decode events and reduced p99 13.1%, but this follow-up's repeated
global-sync controls did not reproduce that exact effect.  That discrepancy is
why single-run results are not used as a policy claim.

## Policy and causal interpretation

* **P1 ALWAYS_SYNC:** useful upper-bound diagnostic only.  It changes global
  scheduling and has measurable throughput cost; it is not a candidate method.
* **P2 ORACLE_SELECTIVE_SYNC:** invocation IDs were selected offline from
  prior runs (one set from dispatch tails and one from wait-proxy+dispatch
  tails).  Local IDs are not a stable cross-run identity, so this is an oracle
  diagnostic rather than a perfect future-aware replay.  It did not provide a
  robust E2E advantage.
* **P3 ONLINE_SIMPLE:** uses only the previous same-process dispatch duration
  and drains the DeepEP communication stream when it exceeds 1 ms.  It
  selected 773/489/roughly comparable rows in the three runs.  Although p99
  sometimes improved, the extreme-tail rate and throughput criteria failed.
* **Relevant stream drain:** run 1 reduced >20 ms events 64→26, but runs 2/3
  had higher p50/tails.  The intervention is not a stable substitute for a
  readiness-aware event wait and can itself serialize progress.

The causal chain remains:

`previous DeepEP async communication remains outstanding`
→ `DeepEP notify/barrier dependency is encountered in the next dispatch`
→ `dispatch CUDA span explodes while expert/combine stay normal`.

The lightweight `run_online.py` driver exposes per-wave `generate()` wall
time and generated-token throughput, but does not expose a per-request TTFT or
TPOT field.  These are therefore recorded as **NOT INSTRUMENTED** rather than
invented.  Wave wall time is a secondary serving proxy; all gate decisions use
same-device CUDA MoE timings and run-level tail rates.

The new evidence also shows the key engineering blocker: a runtime-visible
readiness query for the internal DeepEP communication state is required before
selective intervention can be evaluated fairly.  Moving a wait to an earlier
Python point or draining the whole comm stream is not enough.

## Gates

* Event-wait proxy predicts tails: **YES**, but native readiness is
  unavailable.
* Oracle extreme-tail reduction: **not robust / <50%**.
* Oracle p99 improvement: **not robust / <10%**.
* Throughput loss: **>3%** for the selective runs.
* Online simple recovery of oracle: **not demonstrated**.
* `DOES WAIT MERELY MOVE EARLIER`: **YES for the coarse drain in some runs**;
  dispatch accounting can improve without reliable E2E improvement.

Therefore this is **NO_GO-A/B for a method under the current API**, not a
rejection of the already localized communication-backlog mechanism.  No
production scheduler, RL policy, routing change, or custom kernel was added.

## Next experiment

Expose a nonblocking readiness/query or completion timestamp for the relevant
DeepEP internal event/notify sequence (ideally in the C++ binding), then repeat
paired stock versus event-only wait on a fixed workload.  If the signal can
select only truly outstanding events and recovers the prior diagnostic tail
reduction with ≤2–3% throughput loss, a small wait-aware runtime PoC is
warranted.  Otherwise terminate this direction.
