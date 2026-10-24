# Loop-aware method

Status: `NOT_RUN` for a new loop-aware serving mechanism.
The implemented distributed operator is generic exact context parallelism.
The physical-layer-major allocator control is also generic.
Neither implementation receives a new loop-specific method name.

## Bounded candidate audit

| Candidate | Required property | Current evidence | Consequence |
|---|---|---|---|
| Release earlier-depth KV | Earlier KV becomes dead after its pass. | The next generated token reads that same depth's full causal history. | Releasing it breaks exact inference. |
| Alias depth caches | Original BF16 KV is identical across depths. | Captured depth namespaces have distinct K/V fingerprints. | Similarity does not permit byte-exact aliasing. |
| Prefetch the next depth | A cold memory tier causes a measured residency stall. | Native RLT keeps its full physical pool in HBM. | Copying resident pages adds traffic. |
| Correct depth-dependent shard imbalance | Different depths have different valid context workloads. | Static R4 reads equal valid context lengths at every depth. | Balanced generic context sharding already receives the same information. |
| Physical-layer-major pages | Native strides cause measurable cost. | The exact control changes real strides and receives matched timing tests. | Any benefit remains a generic allocator result. |
| Batch future-loop queries | Future Q values are ready before the current attention. | Hidden-state dependencies prevent this readiness. | Fusing all 96 attention queries is not an exact legal schedule. |

The tiering report charges measured transfer costs.
The scaling report separates operator benefit from full-token benefit.
There is no measured loop-specific improvement over the identical generic operator.
This is not proof that every future exact loop-aware placement method must fail.
It is insufficient evidence for a new loop-specific system claim.

## Integration limit

Native RLT has no matched TP/CP process-group path for Ouro.
Adding a distributed Attention helper requires coordinated graph capture and consistent collective ordering.
Every physical layer and recurrence pass would cross that boundary.
Dynamic admission, cache-page growth, request completion, and graph variants also require coordination.
This exceeds a small allocator or Attention wrapper change.
The present study therefore does not construct a full distributed serving runtime.
It also does not resume the paused Loop PP project.
