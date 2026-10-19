# Native graph argument inventory

## Result

The guarded Base1.4B TP1 probe completed on physical GPU5.
It observed three cached-decode replays from one four-token controlled request.
The retained source graph contains1154 top-level nodes.
The read-only getter inspected1153 kernel nodes.
It read zero argument values.

| Node result | Count |
| --- | ---: |
| Runtime parameter query failed | 865 |
| Name resolved, ABI unknown | 288 |
| Non-kernel node excluded | 1 |

All865 failed queries returned CUDA error98, `invalid device function`.
The288 named kernels comprise96 KV-update kernels,96 FA3 attention kernels, and96 FA3 combine kernels.
An empty explicit ABI policy correctly excluded all argument-value reads.
No code converts a runtime function pointer into a driver function handle.

## Evidence and limits

Raw evidence resides in `outputs/native_graph_argument_inventory_base14_P1_v1/1.4B_R4_TP1`.
Its `argument_observations.json` identifies the observed graph, replay count, and node record path.
The runtime is the Torch-linked CUDA12.9 library.
The guard and the diagnostic status both report success.
Success means that the bounded inventory completed, not that it resolved weight identity.

The result does not establish cached-decode weight operands.
It does not explain the storage counterfactual's numerical differences.
It does not establish executable argument identity, HBM reads, or memory traffic.
It provides no latency measurement or natural-quality result.
Retained source graphs remain a diagnostic condition, not a headline timing baseline.

The previous prefill binding observation remains separate.
It cannot fill this cached-decode evidence gap.
Unknown ABIs and failed queries remain `UNRESOLVED`.
The campaign does not bypass the safety gate to obtain pointer matches.

## Reproduction

The [reader contract](cuda_graph_argument_reader_v1.md) defines opt-in and ABI requirements.
The run command appears in the raw `run_contract.json`.
It uses the existing ownership guard and a new isolated output directory.
The guard restores the scoped utilization worker after completion.
