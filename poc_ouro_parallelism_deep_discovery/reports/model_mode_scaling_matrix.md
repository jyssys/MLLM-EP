# Model, depth mode, and GPU scope

Status: checkpoint. No candidate demonstrates both-size/both-mode architecture speedup.
P8 is unmeasured and unauthorized in this campaign.

## Actual Base targets

| Field | Ouro-1.4B Base | Ouro-2.6B Base |
|---|---:|---:|
| Model revision | `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` | `1ed04250da1a9936042725d302e81c8fa2ab5abd` |
| Physical layers | 24 | 48 |
| Fixed recurrent passes | 4 | 4 |
| Full logical KV slots | 96 | 192 |
| Hidden/intermediate width | 2048/5632 | 2048/5632 |
| Q/KV heads | 16/16 | 16/16 |
| BF16 weight payload | 2,869,305,346 bytes | 5,335,949,314 bytes |
| Full fixed KV per token | 786432 bytes | 1572864 bytes |

Weight payload and KV allocation are capacities, not measured HBM traffic.
The full KV expression is `2*L*R*N*16*128*2` bytes.
The actual native TP layouts shard heads at P2/P4.
The original-gate staged path retains all depth slots under `last_exited`.

## Completed evidence

| Evidence | 1.4B fixed | 1.4B adaptive | 2.6B fixed | 2.6B adaptive |
|---|---|---|---|---|
| Tiny original-gate/state sanity | Measured | Measured | Measured | Measured |
| Natural staged calibration | 16 requests | 16 requests | 16 requests | 16 requests |
| Native TP1 natural calibration | 16 requests | Not supported in this path | 16 requests | Not supported in this path |
| Default native controlled TP1/2/4 | Measured | Not supported in this path | Measured | Not supported in this path |
| Matched native/staged four-position capture | Measured | NOT_RUN | Measured | NOT_RUN |
| Physical P2/P4 MLP and remote control | Measured component | Shared geometry only | Measured component | Shared geometry only |
| Candidate live E2E | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| Candidate held-out quality | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

Component geometry compatibility does not establish adaptive performance headroom.
Native fixed and original-gate adaptive remain distinct baseline paths.
No newly trained CDB lookahead or preloop gate is active.

## Candidate scope after the current screens

| Candidate group | Fixed mode | Adaptive mode | Both sizes | P2/P4 | P8 |
|---|---|---|---|---|---|
| C01/C16 ready unions | Independent requests only without speculation | Mixed depths can supply ready rows | Source and CPU support | Component controls, no new engine | Projection only |
| C02 remote MLP | Tested layout loses | Geometry transfers, online gain untested | Actual weights tested | Actual components | Projection only |
| C03 terminal aliases | No terminal-copy saving | Exact selected-policy opportunity, direct prior | Source and address checks | No candidate live timing | Projection only |
| C04 layouts | Generic algebra/control | Same state obligations plus gates | Both audited | No qualifying new layout | Projection only |
| C05 correction | Four-round exact recovery in bounded probes | Earlier exits shorten the available span | Actual single-token probes | No physical solver | Projection only |
| C07 verified branches | Direct neighboring algorithms | Original-gate matching unresolved | Ownership accounting | No distributed branch executor | Projection only |
| C13 gate overlap | No required exit-gate wait in native fixed mode | Mandatory-prefix overlap, generic baseline work | Source support | No live candidate | Projection only |
| C15 projection-input storage | No useful direct contraction shown | Same representation limits | Geometry and source audit | No new runtime | Projection only |

## P8: capacity checks, not performance

Assume eight future GPUs with the same audited H100 storage limits.
This assumption does not authorize or describe an actual future topology.

| Quantity | Base1.4B | Base2.6B |
|---|---:|---:|
| TP8 dense core/rank | 294 MiB | 588 MiB |
| TP8 logical fixed KV/rank at N8192 | 0.75 GiB | 1.5 GiB |
| Extra full-prefix replication to seven peers at N8192 | 42 GiB | 84 GiB |
| Extra four-position full-R4 KV replication to seven peers | 21 MiB | 42 MiB |

The prefix replication rows describe a particular full-replica design.
Stationary KV can avoid those copies but introduces remote reads/messages instead.
Neither layout receives free state movement.
Shared-prefix references can reduce duplication only where the consumer can access the referenced storage legally.

Even the TP8 dense core exceeds the generous audited on-chip capacity screen.
P8 therefore does not establish complete BF16 core residence.
This does not reject streamed or selectively retained weight tiles.

At C1, a four-depth numerical round contains at most four depth-map tasks in the tested formulation.
P8 cannot create eight independent recurrent passes from that chain.
Additional operator sharding remains possible, but then conventional TP is a necessary control.

## Topology-sensitive projections

The communication fixtures measure P2/P4 on the actual NV18-connected devices.
They report whole-primitive durations, not pure link bandwidth.
They cannot determine an unobserved P8 collective latency.

| Scenario | Explicit topology assumption | Permitted projection | Missing evidence |
|---|---|---|---|
| Optimistic | All future peers have an intra-island path comparable to measured pairs | Use measured pair costs only as candidate-favoring component inputs | Eight-rank startup, contention, and collective algorithm |
| Central | Two four-GPU islands with an unmeasured bridge | `T_bridge = alpha_bridge + bytes/B_bridge`, plus required serialization | Bridge latency/bandwidth and placement |
| Conservative | The bridge serializes critical owner traffic | Add every required critical crossing and state-publication dependency | Actual topology and overlap behavior |

These are parameterized scenario models, not numerical P8 speedup claims.
No projected value can enter a measured-results table.
The campaign must obtain authorization and actual hardware before any P8 validation.
