# Research tree

## Root observation

Previous results show that spectacular route/fanout/tail anomalies carry
little direct request mass.  Source pass 1 additionally shows that the old
observer inserted a layer-local CUDA synchronization into an otherwise async
DeepEP path.  The root question is therefore: **where is repeatable online
critical-path latency mass after removing observer-induced serialization?**

Each completed node must record EXPECTED, OBSERVED, FAILED ASSUMPTION, NEW
SYSTEM FACT, E2E IMPLICATION, and NEXT CHILDREN.

## Initial observation-derived nodes

| ID | Family | Distinct causal hypothesis | Perturbation/control | Headroom test | Status |
|---|---|---|---|---|---|
| H01 | measurement | Per-layer observer synchronization materially serializes async DeepEP | old synchronous hook vs deferred-event hook vs no hook | paired request E2E and throughput | TESTED_CAUTION |
| H02 | measurement | Per-layer route GPU→CPU copies, not CUDA events, dominate observer perturbation | deferred events with route copy off/on | E2E delta and CPU gaps | UNTESTED |
| H03 | DP coordination | Same global work partitioned asymmetrically across DP groups increases request critical path | balanced vs skewed DP assignment, eager, same request multiset | direct wave E2E | UNTESTED |
| H04 | phase alignment | Cross-DP prefill/decode phase mismatch is costlier than token-count asymmetry alone | phase-aligned vs phase-misaligned DP streams | direct wave E2E | UNTESTED |
| H05 | idle participation | A temporarily empty DP group pays a full dummy-model participation tax because EP spans all ranks | useful work on both DP groups vs one group idle | useful-request E2E and GPU steps | UNTESTED |
| H06 | completion churn | Heterogeneous output lengths create repeated small-batch EP waves after peers finish | equal vs mixed max_tokens, equal total output tokens | aggregate E2E/throughput | UNTESTED |
| H07 | global placement | The same request multiset has different cost when heterogeneity is within a DP group vs across DP groups | remap identical requests to DP groups | direct aggregate E2E | UNTESTED |
| H08 | scheduler order | Long-first vs short-first request order changes chunk/EP critical path even with the same multiset | reverse request order | aggregate E2E | UNTESTED |
| H09 | chunk residue | Prompt lengths just above an MBT/chunk boundary create a disproportionately costly residual EP invocation | boundary-minus vs boundary-plus matched useful tokens | residual cost and E2E share | UNTESTED |
| H10 | packing | Few-long vs many-short prompts with equal total scheduled tokens create different per-step non-MoE/MoE mass | composition control | direct E2E | UNTESTED |
| H11 | mixed phase | Co-scheduling prefill and decode serializes otherwise independent work through EP collectives | separated vs overlapped arrival schedule, same requests | direct E2E/ITL | UNTESTED |
| H12 | burst state | Bursty arrivals leave high-frequency runtime debt after matching clocks and current work | steady vs bursty with randomized order | post-burst E2E mass | UNTESTED |
| H13 | DVFS | The earlier state anomaly is GPU clock ramp, not MoE state | clock-cold vs compute-prewarmed, same target | target E2E/telemetry | UNTESTED |
| H14 | TP→EP interaction | At fixed decode M, longer attention context changes following MoE cost via stream/resource state | equal route shape, short vs long KV context | MoE and block E2E | UNTESTED |
| H15 | encoder→EP interaction | Vision encoder residue changes first LLM-MoE critical path after matching LLM token count and clocks | precomputed/text vs live vision control | early-layer/wave E2E | UNTESTED |
| H16 | layer dependency | Layer cost clusters follow preceding attention duration rather than expert/rank load | per-layer paired attention→MoE analysis and injected attention delay | incremental E2E mass | UNTESTED |
| H17 | host metadata | Per-layer expert-token metadata materialization is a high-frequency CPU/GPU serialization point | stock list metadata vs preconditioned/no-copy diagnostic | model-step E2E | UNTESTED |
| H18 | async finalize | Stock async finalize exposes useful inter-layer slack that explicit finalize synchronization destroys | stock vs narrow combine-stream drain | E2E and overlap span | UNTESTED |
| H19 | token order | Token permutation at fixed route histogram changes layout/packing cost independent of load | fixed-route order permutation | dispatch/T_MoE | UNTESTED |
| H20 | DP CPU rendezvous | Per-step CPU DP descriptor all-reduce is a high-frequency critical-path bubble | synchronized vs delayed host arrival with same GPU work | model-step host/GPU gaps | UNTESTED |
| H21 | graph padding | DP imbalance becomes qualitatively worse under CUDA-graph max-rank padding than eager | eager vs graph, balanced/skewed paired | direct E2E and padded tokens | UNTESTED |
| H22 | KV pressure | KV-cache pressure/preemption interacts with EP step synchronization to create removable bubbles | roomy vs constrained KV cache, same stream | direct E2E/throughput | UNTESTED |
| H23 | image-count scheduling | Equal visual-token volume split across one vs multiple images changes encoder/LLM handoff and EP utilization | matched aggregate visual tokens | direct E2E | UNTESTED |
| H24 | non-MoE co-tail | Residual online tails are generic whole-step stalls rather than EP-specific | attention and MoE stage co-tail join | removable E2E mass | UNTESTED |
| H25 | host launch cadence | CPU launch gaps, not GPU stage work, dominate low-concurrency online latency | controlled CPU pacing with same requests | direct E2E/throughput | UNTESTED |
| H26 | first-layer setup | Only early MoE layers pay recurring dynamic-shape setup after scheduler-shape changes | shape transitions with early/mid/late timing | E2E mass and warmup fix | UNTESTED |
| H27 | decode age | Same decode batch width at different context ages changes EP stage mix through upstream attention pressure | age-binned same-M steps | E2E headroom | UNTESTED |
| H28 | request turnover | Admission/completion events create synchronized bubbles independent of current M | steady active set vs turnover with matched M | E2E mass | UNTESTED |
| H29 | DP rank identity | Physical DP placement relative to NVLink topology changes the same logical workload | swap DP-to-physical mapping within GPUs 1–4 | E2E and comm span | UNTESTED |
| H30 | instrumentation storage | Synchronous filesystem/JSON activity in worker instrumentation alters CPU launch cadence | buffered vs per-row logging | E2E/throughput | UNTESTED |
| H31 | async scheduler | Auto-enabled asynchronous scheduling improves host overlap but amplifies DP rendezvous sensitivity | async on/off under same pinned request trace | direct request E2E and DP gaps | SOURCE_DERIVED |
| H32 | DP rendezvous backend | CPU DP descriptor all-reduce avoids GPU sync but becomes the dominant cross-engine phase lock | CPU process-group vs supported NCCL DP sync under otherwise identical async run | E2E, TTFT, host gaps | SOURCE_DERIVED |
| H33 | metadata allocation | Rebuilding pageable CPU and GPU expert-count tensors at every layer is a high-frequency allocation/copy tax | stock metadata path vs storage-reusing diagnostic | direct E2E and host enqueue | SOURCE_DERIVED |
| H34 | async combine copy | The post-combine nonblocking output copy delays the next layer because it is launched only after receiver wait | stock finalize vs preallocated/stream-timed copy diagnostic | layer critical path and E2E | SOURCE_DERIVED |
| H35 | process-group progress | OS scheduling variance of the CPU DP collective, rather than GPU work, creates repeatable cross-rank bubbles | fixed GPU work with controlled host skew/affinity | step and request E2E | SOURCE_DERIVED |
| H36 | idle dummy participation | Removing the last short request from an otherwise identical DP rank changes the dummy-participation tax independently of local batch geometry | six long requests on DP0 with one/two short participants versus no DP1 request | direct wave E2E and dummy markers | SOURCE_DERIVED |
| H37 | completion spread | Equal aggregate decode work but heterogeneous per-request lengths inflate wave completion/throughput even when request median is unchanged | exact prompt pool and output sum, uniform versus heterogeneous lengths | request p99, last completion, tokens/s | RETHINK_5_PENDING |
| H38 | metadata tax | Per-layer host metadata materialization is a recurrent, measurable execution fraction | instrument `make_from_list`, storage-reuse diagnostic | direct step E2E | RETHINK_5_PENDING |
| H39 | phase kernel regime | Phase placement changes kernel/packing regime at matched work, explaining a sign-reversed H04 micro-effect | same work/route, kernel identity and shape-transition control | stage and E2E | RETHINK_5_PENDING |
| H40 | whole-GPU co-tail | Attention and MoE tails share a generic GPU resource event, not EP debt | matched shape co-tail and text-only control | direct E2E mass | RETHINK_5_PENDING |
| H41 | response semantics | H06 wave effect is an API completion artifact rather than model execution | server throughput/step trace versus request completion | causal E2E | RETHINK_5_PENDING |

## Completed-node records

### H01 — deferred observer trust gate

- EXPECTED: removing per-layer synchronization and route copies should make the
  observer statistically indistinguishable from no-hook serving.
- OBSERVED: after discarding warmup, no-hook/deferred request E2E medians were
  1544.54/1595.01 ms (+3.27%); TTFT medians were 178.92/181.82 ms (+1.62%).
- FAILED ASSUMPTION: deferred event creation itself is not literally free at
  15 sampled stage boundaries per layer-set and four workers.
- NEW SYSTEM FACT: buffered, nonblocking event observation preserves the
  qualitative online regime but introduces a small measurable E2E tax.
- E2E IMPLICATION: absolute latency comparisons across server launches require
  no-hook controls; within-server randomized A/B effects above 5% remain
  interpretable, while 2–5% effects need a replication without the observer.
- NEXT CHILDREN: H02 separates event versus route-copy cost; H30 tests logging
  cadence; all runtime hypotheses use randomized within-server A/B first.

## Resumed live controls (2026-09-07)

### H36 — idle DP dummy participation (fresh worker)

- EXPECTED: removing the last short request from DP1 would expose a large
  dummy-model participation tax or an idle-rank critical-path effect.
- OBSERVED: corrected `Worker.execute_dummy_batch` instrumentation recorded
  6,399 dummy calls on each DP1 worker and 286 on each DP0 worker. The
  randomized A/B request-wave median was 2,304.57 vs 2,298.62 ms (-0.26%);
  request median was -0.25%.
- FAILED ASSUMPTION: an idle DP rank is silent or necessarily lengthens the
  useful DP rank's wave.
- NEW SYSTEM FACT: DP workers do execute explicit dummy batches, but this
  participation was not a material latency driver for this workload.
- E2E IMPLICATION: H36 is a measured negative; the prior large wave-level
  completion effect must be conditioned on request semantics or another state
  variable, not simply idle participation.
- NEXT CHILDREN: retain dummy markers for future scheduler-state controls;
  do not promote idle-DP policy without a new high-mass signal.

### H31 — asynchronous scheduler control

- EXPECTED: disabling asynchronous scheduling would remove or strongly reduce
  the heterogeneous-wave divergence seen in H06/H47.
- OBSERVED: with `--no-async-scheduling`, DeepEP HT remained active and the
  A/B request-wave median differed by -1.90% (request median -1.83%); no large
  completion-spread split appeared.
- FAILED ASSUMPTION: the prior 80--90% wave split is caused by the async flag
  alone.
- NEW SYSTEM FACT: async scheduling is a real runtime switch, but this paired
  control does not reproduce the anomaly.
- E2E IMPLICATION: async scheduling is not sufficient as a standalone
  explanation; the H06/H47 wave result is likely composition/worker-state
  conditional.
- NEXT CHILDREN: H32 tests DP synchronization backend; H37/H46 retain
  completion-spread and turnover as conditional hypotheses.

### H32 — DP synchronization backend control

- EXPECTED: forcing NCCL for DP synchronization would expose or suppress a
  cross-engine rendezvous effect relative to the CPU/Gloo default used with
  async scheduling.
- OBSERVED: `--no-disable-nccl-for-dp-synchronization` logged
  `disable_nccl_for_dp_synchronization: False`, with async scheduling and
  `DeepEPHTPrepareAndFinalize` active. The matched A/B request-wave median
  differed by -0.01% (request median -0.03%).
- FAILED ASSUMPTION: the DP synchronization backend determines the tested
  request-shape contrast.
- NEW SYSTEM FACT: the NCCL DP path is supported in this environment but did
  not produce a measurable paired effect in the H32 workload.
- E2E IMPLICATION: H32 is a measured negative; any remaining anomaly is not a
  simple CPU-vs-NCCL DP backend choice.
- NEXT CHILDREN: a narrower host-rendezvous trace would be needed only if a
  future high-mass residual co-occurs with DP phase skew.

### H38 — metadata/step recurrence (fresh follow-up)

- EXPECTED: repeated metadata construction and shape history would produce a
  stable model-kernel effect after controlling request shape.
- OBSERVED: the prior v5 run produced an extreme A/B wave split, but its A/B
  output budgets were not equivalent; metadata instrumentation on 287,219
  calls measured 0.1137 ms median, 0.125 ms p90, 0.145 ms p99 and a 13.9 ms
  maximum.
- FAILED ASSUMPTION: a recurrent host metadata path automatically creates a
  high-mass request-level opportunity.
- NEW SYSTEM FACT: metadata construction is frequent and normally tiny, with
  rare host outliers; its direct E2E mass remains unproven.
- E2E IMPLICATION: H38 is a diagnostic/measurement candidate, not a method
  candidate unless fresh v7 data shows a reproducible request-level tail.
- NEXT CHILDREN: v7 H33 contrasts metadata-heavy and metadata-light request
  shapes; inspect only high residual calls.

### v7 closure — source-derived and state controls

The final fresh worker campaign collected 21,456 successful requests and
59,789 logical sampled steps under the same BF16 TP2/DP2/EP4 DeepEP-HT path.

#### H33 — metadata host-path control

- EXPECTED: metadata-heavy versus metadata-light request schedules would
  change host/device metadata materialization and request latency.
- OBSERVED: request p50 changed +1.73%, while wave p50 changed +91.78%; the
  A/B request output budgets and completion spread are not equivalent. The
  frequent metadata call remains typically ~0.11 ms from the prior direct
  observer.
- FAILED ASSUMPTION: a wave split with different decode work identifies a
  metadata tax.
- NEW SYSTEM FACT: the fresh v7 host-path contrast is not a valid causal
  metadata comparison.
- E2E IMPLICATION: no metadata optimization headroom is established.
- NEXT CHILDREN: only a no-output-budget-change metadata reuse experiment
  would be informative; deprioritized after the negative H36/H43 controls.

#### H43 — pinned versus unpinned DP

- EXPECTED: supported DP pinning could create a scheduling or rendezvous
  artifact.
- OBSERVED: 330/333 steady waves; request p50 B-vs-A = -0.15% and sampled
  MoE = -0.19%.
- FAILED ASSUMPTION: request-to-DP pinning is a material source of the prior
  state anomaly.
- NEW SYSTEM FACT: pinning is effectively null for this workload.
- E2E IMPLICATION: close as NO_GO.
- NEXT CHILDREN: none without a new host-affinity causal variable.

#### H45 — common regime transition repeat

- EXPECTED: the chunk-residue repeat would separate A and B after randomized
  ordering.
- OBSERVED: 562/561 waves; request p50 B-vs-A = -0.12%, while both variants
  share large common-mode wave variation (CV 0.22/0.15).
- FAILED ASSUMPTION: the earlier common transition was treatment-specific.
- NEW SYSTEM FACT: state drift is reproducible as a common-mode transition,
  not a useful A/B lever.
- E2E IMPLICATION: anomaly is diagnostic only; no method headroom.
- NEXT CHILDREN: a clock/allocator trace would characterize it, but it is not
  a high-mass causal intervention in this sprint.

#### H46 — turnover repeat

- EXPECTED: request-set turnover would produce a stable request-level penalty.
- OBSERVED: wave p50 B-vs-A = +89.62%, but request p50 = -2.30%; both variants
  also entered the same elevated common state.
- FAILED ASSUMPTION: wave completion spread represents per-request latency.
- NEW SYSTEM FACT: cohort completion semantics can dominate wave metrics while
  leaving request medians unchanged or slightly improved.
- E2E IMPLICATION: reject as an optimization target; retain as measurement
  caution.
- NEXT CHILDREN: none in MoE runtime without a throughput-specific question.

#### H48 — mixed-phase repeat

- EXPECTED: a mixed prefill/decode schedule would expose a stable phase-order
  effect.
- OBSERVED: wave p50 B-vs-A = -17.13%, request p50 = -0.29%, sampled MoE =
  -2.05%; A/B remain in the same ~10 s common state.
- FAILED ASSUMPTION: wave-level phase separation implies a request-level
  MoE opportunity.
- NEW SYSTEM FACT: mixed-phase tails are dominated by shared serving state in
  this path.
- E2E IMPLICATION: close as NO_GO.
- NEXT CHILDREN: none; only a new non-MoE resource trace could change the
  interpretation.
