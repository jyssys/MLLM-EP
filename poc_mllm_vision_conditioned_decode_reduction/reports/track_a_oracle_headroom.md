# Track A — Oracle headroom

Status: **MEASURED_BOUNDED_VECTOR_ORACLE**

The frozen long-reasoning route traces contain expert IDs and router weights, but no per-path expert output vectors. Route overlap alone cannot establish whether re-routing is semantically safe.

## Capture integrity

- Shards: 6017
- Complete top-8 token/layer groups: 1530
- Unmatched receiver rows: 0
- Ambiguous receiver rows: 0
- Incomplete groups: 0
- Discarded non-decode router rows: 1200
- Receiver shards with filename/canonical-step mismatch: 0
- Receiver rows with filename/canonical-step mismatch: 0

## Bounded oracle frontier

| Requested path reduction | Actual paths | Active-union reduction | Mean output cosine | P90 weighted contribution rel-L2 | P90 mixture rel-L2 |
|---:|---:|---:|---:|---:|---:|
| 10.0% | 10.00% | 9.59% | 0.538555 | 0.113274 | 0.324954 |
| 20.0% | 20.00% | 19.14% | 0.344683 | 0.170652 | 0.492585 |
| 30.0% | 30.00% | 28.07% | 0.251862 | 0.211843 | 0.651425 |
| 40.0% | 40.00% | 37.51% | 0.202885 | 0.250542 | 0.808516 |

Stage-A local-vector gate: **FAIL**.

This oracle is not a full-model quality result and cannot authorize timing unless the separate natural-generation quality frontier also passes.
