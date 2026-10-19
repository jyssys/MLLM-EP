# P8 projection feasibility supplement, version 1

Status: `PARTIAL_DISCLOSURE_ONLY`.
P8 remains `PROJECTED_NOT_RUN`.

## Result

The current dossier names optimistic, central, and conservative topology scenarios.
It does not fully satisfy the specification's calibrated P8 projection requirement.
No candidate has a complete P8 ownership schedule, calibrated critical path, or matched native TP8 reference.
This supplement adds explicit component equations and capacity accounting.
It does not supply unsupported latency values.

Specification Section 16.4 requires projections from actual P2/P4 behavior and a stated topology.
The [current scope report](model_mode_scaling_matrix.md:79) supplies scenario names and unknown bridge parameters.
The [cost script](../scripts/cost_screens.py:10) evaluates measured P2/P4 inputs.
It does not evaluate P8.
The [headroom synthesis](../artifacts/current_headroom_decomposition_v1.json) still marks candidate paths incomplete.

This audit preserves those limits.
It adds no GPU access, architecture implementation, candidate result, or change to prior artifacts.

## What the closed hardware evidence says

The archived [environment manifest](../artifacts/environment_manifest.json) lists NV18 between every GPU0–7 pair.
Its legend defines NV18 as 18 bonded NVLinks.
Actual authorized experiments use physical GPUs4–7.
The P2/P4 fixtures do not exercise the other four GPUs.

A future same-host eight-rank scenario can conditionally use that archived connectivity.
It cannot infer achieved eight-rank bandwidth, concurrent peer behavior, or authorization.
The archive does not establish the topology of another future machine.
This audit makes no NVSwitch hardware claim.

The existing “central” scenario assumes two four-GPU islands and an unmeasured bridge.
That is an alternate hypothetical topology.
It is not an evidence-derived central estimate for the archived all-NV18 table.
The label “central” is not a statistical median.

## Calibration that exists

The communication fixture uses NCCL broadcast and all-reduce on zero-valued BF16 payloads.
It captures 16 repetitions per graph and records 31 trials after warm-up.
The median uses each trial's maximum rank-local stream duration.
It is a whole-primitive interval, not pure link bandwidth.

| Payload | P2 broadcast, µs | P4 broadcast, µs | P2 all-reduce, µs | P4 all-reduce, µs |
|---:|---:|---:|---:|---:|
| 4096 B | 10.350 | 8.420 | 12.258 | 13.414 |
| 786432 B | 12.274 | 13.782 | 15.178 | 18.668 |
| 1572864 B | 24.162 | 31.040 | 21.022 | 24.890 |
| 67108864 B | 204.946 | 220.042 | 255.702 | 330.036 |

Raw evidence: [P2](../outputs/communication_P2_v1/communication.json), [P4](../outputs/communication_P4_v1/communication.json), and [probe source](../scripts/communication_probe.py).
The [JSON supplement](../artifacts/p8_projection_feasibility_supplement_v1.json) retains selected trial ranges and exact medians.

P2 communication uses physical4/5.
P2 MLP measurements instead use physical4/6.
P4 uses physical4/5/6/7.
The P4 remote-MLP test still communicates only between ranks0/1.
It is not an all-owner four-rank exchange measurement.

P4 broadcast is faster than P2 at 4096 B in these separate runs.
Thus, a simple monotone latency-per-rank fit already lacks support.
The fixtures also use a different primitive from native custom all-reduce.
Warm operator measurements do not establish whole-stack cache reuse or interference.

The [controlled native table](../analysis/checkpoint_03/native_summary.csv) provides actual P1/P2/P4 fixed-R4 timing.
It uses two controlled request shapes and sequential method order.
It does not identify compute, communication, and overlap terms uniquely.
It cannot determine a P8 point by itself.
Different later teacher/output protocols must not enter an unconditional scaling ratio.

## Conditional P8 component scenarios

Let `b2(M)` and `b4(M)` denote the measured broadcast medians above.
Let `beta(M)=alpha_bridge+M/B_bridge` describe a hypothetical bridge.
This expression is a model term, not measured bridge service.
Every scenario also needs graph, queue, completion, and version-publication costs.

| Scenario | Explicit assumption | Conditional component model | Remaining blocker |
|---|---|---|---|
| Optimistic | Future same-host NV18 paths sustain P4-like behavior at eight ranks | `b8(M)=b4(M)+Delta8(M)` | No calibration bounds `Delta8`. Setting it to zero is only a favorable sensitivity assumption. |
| Central | Two four-rank islands use a serialized local/bridge/local broadcast | `b8(M)=2*b4(M)+beta(M)` | Bridge latency, bandwidth, contention, and required communicator are unknown. |
| Conservative | Seven prepared peer deliveries execute serially | `b8(M)=sum_j b2_j(M)+control` | The archive measures one pair, not all seven delivery costs. |

These are component models, not candidate or native TP8 latency predictions.
They are not guaranteed ordered outcomes.
Different protocols and topology assumptions can change their ordering.

For the central model, the calibrated local terms are 16.840, 27.564, and 62.080 µs at the first three table payloads.
Each value still needs its unknown bridge term.
For a separate seven-identical-pair sensitivity model, the corresponding values are 72.450, 85.918, and 169.134 µs.
Those values assume every prepared pair behaves like the measured pair.
They exclude setup, contention changes, and control.
They are neither measured P8 values nor latency upper bounds.

These equations do not convert one broadcast into an all-reduce, all-gather, reduce-scatter, or remote-attention path.
A candidate must specify its actual primitive and causal message count.
Summing measured medians creates a model, not an observed request path.
Full-prefix transfers also exceed the fixture's largest 64 MiB message.
Chunking, overlap, and large-transfer cost remain unknown.

## Native TP8 can change the primitive

The installed [custom-all-reduce source](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/distributed/device_communicators/custom_all_reduce.py:51) lists world size8.
Support is conditional.
The constructor checks same-node placement, full connectivity for more than two ranks, and software P2P access.
A cross-node or unsuitable island layout can disable this path.
The matched native control could then use a different communication implementation.

Therefore, the central scenario cannot simply add bridge time to the measured native P4 custom-AR path.
The component NCCL fixture does not resolve that change.
No installed constructor or P2P check executes in this audit.

## Ownership and capacity remain separate

Both Base models use D2048, I5632, 16 MHA heads, and width128.
P8 divides those dimensions into two heads and 704 FFN channels per rank.
This arithmetic does not establish a fast kernel or a complete native TP8 run.

Balanced dense-core ownership gives 294 MiB/rank for Base1.4B and 588 MiB/rank for Base2.6B.
These counts cover the seven dense matrices, not all embeddings, norms, buffers, or allocator reservation.
The full physical core remains one aggregate copy under ordinary TP.
Four recurrent visits do not require four weight copies.

For fixed R4, `K=2*L*4*16*128*2` bytes per historical position.
Expanded `last_exited` retains the same slot capacity even when adaptive execution skips work.
The following rows assume N8192 and no history sharing between independent requests.

| Base / system C | TP8 history/rank | Full-history replica history/rank |
|---|---:|---:|
| 1.4B / C1 | 0.75 GiB | 6 GiB |
| 1.4B / C8 | 6 GiB | 48 GiB |
| 2.6B / C1 | 1.5 GiB | 12 GiB |
| 2.6B / C8 | 12 GiB | 96 GiB |

The scope report's 0.75/1.5 GiB P8 history values are per request.
They are not C8 totals.
The complete JSON retains C1/2/4/8 and N1024/4096/8192.

Assume each future device has the measured pool's 85028372480-byte capacity.
Base2.6B C8 full-history replication already exceeds that capacity before weights and scratch.
This rejects that allocation, not TP8, request ownership, or all distributed branch designs.
A full replica also retains 2869305346 or 5335949314 weight bytes.
Private branch state, graph memory, and communication buffers remain additional.

For one initially owned C1/N8192 prefix, full replication to seven peers sends 42/84 GiB of logical payload.
Publishing one complete new fixed-R4 position to those peers sends 5.25/10.5 MiB.
Stationary or sharded KV can change these messages.
It must charge its own consumer access and publication instead.
Shared parameter identities do not make different-depth KV interchangeable.

## Ready work and complete critical paths

Non-speculative decode supplies at most C ready token fronts, not four fronts per request.
At C1/2/4/8, an owner-per-stage P8 design has at most 1/2/4/8 independent request tasks.
Native TP8 can already cooperate across all ranks on one ready row.
Prefill chunks and verified speculative branches need separate readiness contracts.

For a fixed-R4 token, the ordinary two-reduction-per-layer layout has 192/384 reduction calls.
A balanced cyclic physical-layer pipeline instead has 31 core boundaries at P8.
Neither count predicts latency.
The payload, active ranks, local kernel shapes, overlap, startup/drain, and embedding/head placement differ.

Adaptive execution also needs actual original-gate decisions and terminal-state publication.
Mean depth alone cannot scale all costs or allocated KV.
No complete adaptive P8 calibration exists for either Base size.

## Concrete blockers

1. Select the future topology and participant mapping. The current three scenario labels do not do this.
2. Define a candidate-specific ownership and causal message ledger. Preserve startup, drain, rejected work, and publication.
3. Calibrate the matched P8 native primitive and candidate kernels, or retain their explicit unknown terms.
4. Quantify interference, scheduling, graph buffers, and total allocation under the actual ready frontier.
5. Check prediction error before interpreting a small projected advantage.

The no-P8-access constraint leaves those measurements unavailable here.
This supplement documents the gap instead of filling it with advertised bandwidth.
The requirement remains incomplete for a calibrated candidate-level P8 forecast.
No P8 value belongs in a measured-results table.

## Actual work

Work started at 2026-10-19T18:26:18Z.
The JSON records the final accounting cutoff.
No pause, deliberate wait, GPU access, or model execution occurred.
This interval overlaps coordinator wall time.
It does not add independent campaign elapsed hours.
