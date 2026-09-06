# Experiment log

## 2026-09-06 — preparation

- Working contract created and read in full.
- Existing reports audited; closed directions recorded in `KNOWN_NEGATIVE_SPACE.md`.
- Source-discovery pass 1 inspected vLLM 0.20 DP coordination, DBO streams, and
  DeepEP HT prepare/finalize without choosing a target method.
- Found that the inherited observer synchronizes every layer; H01/H02/H30 are
  mandatory measurement-trust gates before all other live results.

## 2026-09-07 — measurement trust and source-discovery pass 2

- Verified physical GPUs 1–4 were occupied only by the user-authorized
  `run_utilize.sh` tree, terminated exactly that tree, and left GPUs 0/5/6/7
  untouched.
- Live no-hook versus deferred-event H01 control (after warmup): request E2E
  medians 1544.54 versus 1595.01 ms (+3.27%); TTFT 178.92 versus 181.82 ms
  (+1.62%). The observer is `CAUTION`, not clean enough for absolute baseline
  claims; randomized within-server comparisons remain usable.
- Deferred observer recorded same-device events without errors: 4,864 worker
  steps in the 121-second smoke run, with representative layers 0/12/24/36/47.
- Discarded an initial H04 block after an audit found unequal prompt work. The
  workload generator was corrected so H03/H04/H05/H11 now preserve total work
  or the exact request multiset, depending on the causal question; protocol
  `v2_matched` separates valid rows.
- Source-discovery pass 2 read current vLLM 0.20 scheduler, GPU runner, DP
  coordinator, and DeepEP HT prepare/finalize without selecting a result in
  advance. Unexpected facts: (1) eager DP still performs a CPU-process-group
  all-reduce every model step, (2) async DeepEP dispatch is followed by a
  receiver event wait, (3) expert-token metadata has an acknowledged GPU/CPU
  materialization path, and (4) DP padding is avoided in eager/no-DBO but the
  rendezvous remains. These facts independently motivate H17/H20 and make
host-arrival skew a distributed critical-path variable worth measuring.

## 2026-09-07 — resumed fresh-worker controls

- Restarted the validated Qwen3-VL TP2/DP2/EP4 DeepEP HT server with the
  corrected `.venv` executable. An initial restart with the unrelated
  anaconda executable failed at startup because its DeepEP kernel package was
  absent; no data from that failed attempt was used.
- H36 fresh worker: corrected `Worker.execute_dummy_batch` hook captured
  explicit idle-DP RPCs (6,399 per DP1 worker, 286 per DP0 worker). Randomized
  A/B request medians differed by only -0.26%, closing idle participation as a
  high-mass explanation.
- H31 source control: `--no-async-scheduling` was accepted and logged by both
  engine cores. DeepEP HT stayed active; request-wave effect was -1.90% and
  request median -1.83%, with no large completion-spread divergence.
- H32 source control: `--no-disable-nccl-for-dp-synchronization` was accepted,
  logging `disable_nccl_for_dp_synchronization: False` while async scheduling
  and DeepEP HT remained active. A/B request-wave effect was -0.01% and
  request median -0.03%.
- The corrected analyzer now reconstructs hypothesis/variant/pair identity
  from worker request IDs when context-file timing loses the client context;
  this is an identity-only fallback and does not alter CUDA execution.
- Final fresh-worker v7 campaign completed H38/H43/H33/H45/H46/H48 with
  21,456 successful request rows and 59,789 logical sampled steps. H43 and
  H45 were request-level null (about -0.15% and -0.12%); H33/H38/H46 retained
  output-length/completion-spread wave effects but no qualifying request
  effect; H48 was also request-null. This campaign adds 3,600 live seconds
  (4.0 four-GPU hours) and closes the remaining frontier.
