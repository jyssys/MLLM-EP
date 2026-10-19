# Actual native graph topology

Status: bounded topology diagnostic, not a candidate implementation.
The job uses pinned Base Ouro-1.4B, BF16, fixed R4, and native asynchronous scheduling.
It runs on physical GPU5.
The runtime retains the actual replayed CUDA graph and exports its DOT description.
It does not replace a model operator.

## Observed graph

The request uses the frozen 1024-token calibration shape and four teacher output tokens.
The observed graph replays three times.
Its [raw topology record](../outputs/native_graph_dag_base14_P1_v4/topology.json) preserves the checkpoint revision and worker capture.

| Quantity | Observed value |
|---|---:|
| Graph nodes | 1154 |
| Kernel nodes | 1153 |
| DtoD memcpy nodes | 1 |
| Edges | 1153 |
| Current-KV write kernels | 96 |
| Attention-core kernels | 96 |
| Attention split-merge kernels | 96 |
| Maximum ready nodes in the captured DAG | 1 |

The [CPU parser](../scripts/parse_native_graph_dot.py) verifies a single directed chain.
Its [analysis](../analysis/native_graph_topology_v1/topology_summary.json) retains node classifications and the raw-file hash.
Five CPU parser tests pass.
An earlier parser attempt omitted the differently formatted memcpy node.
The parser now includes that node and rejects unknown edges.
No result from the failed parse became a latency claim.

## Interpretation boundary

This graph records execution order, not a proof that every stream edge is mathematically necessary.
Its node counts do not measure kernel duration.
The single-chain result does not prove that no transformed exact or verified algorithm can expose useful concurrency.
It does show that this native capture does not already contain four independent recurrent passes.

Observed kernel-pattern cuts occur at positions 0, 288, 576, 864, and 1154.
Boundary kernels can fuse preceding-pass normalization with next-pass input work.
Those cuts are not exact timestamps for Python recurrent-pass completion.
The trailing 4096-byte DtoD copy has no attributed purpose in this checkpoint.
Its size alone does not establish a removable bottleneck.

## Measurement limits

Graph retention, debug mode, and output capture belong to the diagnostic lane.
This job supplies no headline E2E timing or candidate speedup.
The four-position numerical comparison is a separate artifact.
Natural output parity, benchmark accuracy, adaptive execution, and distributed graph topology remain untested by this job.

## Reproduction

```bash
/home/esjung/.venvs/llada2-mini-reference/bin/python \
  -m poc_ouro_parallelism_deep_discovery.scripts.parse_native_graph_dot \
  --dot poc_ouro_parallelism_deep_discovery/outputs/native_graph_dag_base14_P1_v4/rank0_replayed_graph0.dot \
  --layers 24 --output /absolute/new/analysis/directory
```

The parser refuses an existing output directory.
The guarded runtime command remains in the job's `run_contract.json`.

## Later edge-semantics warning

A later live native query returned `cudaErrorLossyQuery` when it omitted CUDA edge metadata.
The endpoint-only chain above therefore does not establish full kernel-completion serialization.
CUDA edges can specify launch or programmatic readiness rather than complete producer termination.
The endpoint counts remain observed facts.
The reported ready-node count is only an endpoint-graph property, not a physical concurrency limit.
Actual edge ports/types and any overlap duration require separate inspection.
No event diagnostic will replace those edges with full fences to obtain a headline timing.
