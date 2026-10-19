# Full-core storage counterfactual results

## Later validity warning

The global AOT model cache remained shared in these nominally isolated v2 runs.
The saved top-level weight operands do not consistently match the requested alias/copies arm.
Actual physical-storage execution is therefore unvalidated for every timing row below, including P2.
P2 numerical equality does not resolve this provenance problem.
The table remains a record of observed jobs, not an interpretable residency comparison.
Future v3 runs must isolate `VLLM_CACHE_ROOT` before imports and inspect actual compiled operands.
The earlier tentative P2 interpretation is withdrawn.
This warning does not identify the cause of P4 numerical drift.

## Scope

These are fixed-R4, B1 controlled-work native runs.
They are not a new architecture or natural-quality evaluation.
Each arm retains all four equal-valued block-parameter copies before timing.
The alias arm executes the original core four times.
The copies arm executes one physically distinct core per depth.
Attention objects, KV slots, parameter-free RoPE, embeddings, loop norm, and output head retain the declared ownership.

The new arms use independent job-local vLLM compilation-cache directories.
This isolates that cache, not every lower-level compiler cache.
Each cell contains two frozen request shapes with five repeats per shape.
The workload generates 128 teacher-forced tokens after the stated prompt length.
Configuration order is sequential.
Two shapes do not establish broad request-level confidence.

## Clean controlled E2E

Each entry gives alias / copies median wall latency in ms.
The final column gives copies latency divided by alias latency across the three contexts.
A value above 1.000× favors alias execution.
No column represents a candidate speedup.

| Model | TP | N1024, ms | N4096, ms | N8192, ms | Copies / alias ratios |
|---|---:|---|---|---|---|
| 1.4B | 2 | 928.389 / 927.949 | 1006.298 / 1006.690 | 1152.165 / 1153.259 | 0.9995× / 1.0004× / 1.0010× |
| 1.4B | 4 | 793.508 / 791.512 | 862.288 / 838.786 | 936.305 / 934.089 | 0.9975× / 0.9727× / 0.9976× |
| 2.6B | 2 | 1821.855 / 1822.220 | 1986.001 / 1987.261 | 2279.291 / 2238.670 | 1.0002× / 1.0006× / 0.9822× |
| 2.6B | 4 | 1572.229 / 1567.011 | 1655.370 / 1597.044 | 1840.376 / 1727.859 | 0.9967× / 0.9648× / 0.9389× |

The tested shared-core execution shows no consistent latency advantage.
The P4 rows cannot establish a pure physical-sharing effect because their numerical and lowering equivalence remains unresolved.
The table does not prove that shared weights are harmful or never useful.

## Numerical checks

P2 alias/copies captures match bitwise for logits, hidden state, and sampled KV in both sizes.
Those captures contain four controlled positions and selected cache features.
They do not establish full natural-output or benchmark parity.

P4 captures differ numerically despite independent per-job vLLM caches.
The full captured-logit relative L2 is 0.010243 for 1.4B and 0.014770 for 2.6B.
Sampled KV relative L2 ranges across ranks from 0.006934 to 0.007607 for 1.4B.
It ranges from 0.008639 to 0.009596 for 2.6B.
All four captured top-1 choices match in every arm.
No post-hoc tolerance converts those P4 differences into an equivalence PASS.

The source audit compares graph interfaces and lowering separately.
Equal intended algebra and equal parameter words do not prove equal compiled arithmetic.
No hardware counter establishes cache hits or actual HBM traffic.

## Capacity controls

Both arms have equal unique parameter-storage bytes at each model/TP point.
Both also have equal recorded allocated and peak allocated bytes at each point.
These are capacity measurements, not traffic measurements.
The 2.6B P4 allocator reserves slightly different amounts despite equal allocated bytes.
All original per-rank memory fields remain in the JSON.

| Model | TP | Unique parameter storage per rank, bytes | Peak allocated per rank, bytes |
|---|---:|---:|---:|
| 1.4B | 2 | 5135407106 | 26923559424 |
| 1.4B | 4 | 2568492034 | 24393868800 |
| 2.6B | 2 | 10069481474 | 31882865152 |
| 2.6B | 4 | 5036315650 | 26943677952 |

## Interpretation and provenance

The P2 counterfactual does not support a large full-stack pointer-reuse benefit in these conditions.
The P4 numerical difference remains a separate uncertainty.
The result does not reject selective tile retention, a different legal frontier, or memory-capacity uses of weight sharing.
It supplies no adaptive-mode result.

The current result file is `artifacts/storage_isolated_checkpoint3.json`.
The raw jobs have the suffix `isolated_v2`.
The earlier shared-cache v1 results remain `CONFOUNDED` for residency attribution.
See [storage_numerical_audit.md](storage_numerical_audit.md) and the later source-level P4 audit when available.
The implementation contract is [full_core_alias_control_contract.md](full_core_alias_control_contract.md).
