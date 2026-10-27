# Runtime audit

Status: `PASS_AUDIT`. The 128/32-question baseline pilot is complete.
The user deferred full benchmark reproduction.
Evidence: `../results/refinement_aware_ep_expert_allocation/environment.json`.
The authoritative contract is `refinement_aware_ep_expert_allocation_poc_spec.md`.
The campaign read all 356 specification lines before implementation.

## Scope

The model is `inclusionAI/LLaDA2.0-mini`, not `mini-preview`.
The checkpoint revision is `dad945cac317da394b390f82c7b40691d8a881ed`.
The branch is `flashvep/refinement-aware-ep-expert-allocation-poc`.
The initial repository commit is `429fb6621e0267b68fabef458c6554aafec90256`.
The previous Looped MoE campaign remains stopped.
This campaign does not modify its artifacts or verdicts.

## Hardware and model

| Item | Verified value | Evidence class |
|---|---|---|
| GPU count | 8 H100 GPUs with 80 GB HBM each | Hardware query |
| Peer topology | NV18 between all GPU pairs | Hardware query |
| NUMA node | 0 for all GPUs | Hardware query |
| Group A | Physical GPUs 0,1,2,3 | Runtime configuration |
| Group B | Physical GPUs 4,5,6,7 | Runtime configuration |
| Transformer layers | 20 | Checkpoint config |
| Dense first layers | 1 | Checkpoint config |
| Routed MoE layers | 19 | Actual module inventory |
| Routed experts per MoE layer | 256 | Checkpoint config |
| Router top-k | 8 | Checkpoint config and dispatch assertions |
| Shared experts per MoE layer | 1 | Checkpoint config |
| Hidden size | 2,048 | Checkpoint config |
| Expert intermediate size | 512 | Checkpoint config |
| Routing | Sigmoid, correction bias, normalized selected weights, scale 2.5 | Checkpoint and source audit |
| Precision | BF16 weights and activations | Runtime inventory |

`environment.json` records every GPU UUID and source hash.
It also records CPU, memory, driver, and source revisions.
Per-rank ownership files record Torch, CUDA, NCCL, and package versions.
The ownership files reside under `ep4/A_sanity/`.

The validated native path uses dense TP4 and source sequence sharding.
Its MoE path uses physically sharded expert weights and DeepEP dispatch/combine.
This configuration is TRUE EP4, not TP4 alone.
It is not a pure EP-only dense-layer configuration.

## Resource control

The resource manager verifies UID, command path, session ownership, and burner log destination.
It does not signal unrelated GPU jobs.
It stopped the owned eight-GPU burner supervisor before group allocation.
It restored temporary idle GPUs during smoke execution.
The durable baseline queue restores owned burners after each GPU group finishes.

Concurrent HF and EP4 runs provide quality measurements only.
They do not provide isolated primary latency results.
Primary latency must exclude all other-group jobs, including burners.
The primary isolated latency field is `NOT_MEASURED`.

## Reuse and failures

The EP4 observer comes from commit `dc539d3222193ab0902e0a342679edc3610f54bb`.
The HF helper comes from commit `c02563979fe818a93c3f6f64b162cdf3df69a8e2`.
The active dInfer source includes an existing DeepEP bridge patch.
The exact working-file hashes identify that patch.
The initial driver source snapshot was not captured at process launch.
The later checkpoint does not establish its exact initial in-memory bytes.
New worker starts save driver snapshots and source hashes.
The current driver changes concern logging, observation, and baseline repetition controls.

The initial preparation command found no `datasets` package in the HF environment.
The corrected preparation command reads cached Arrow files with the CPU environment.
The campaign installed no packages into either validated model environment.

No OOM occurred in the completed smoke tests.
Larger batch sizes remain `NOT_RUN`.
The native kernel reports a missing tuned H100 configuration for its exact expert geometry.
It uses the existing default BF16 Triton configuration.
This limitation must remain visible in performance interpretation.
