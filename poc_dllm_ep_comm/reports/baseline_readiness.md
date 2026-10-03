## 1. Model, runtime, and prior boundary

Selected the already-cached **inclusionAI/LLaDA2.0-mini**, revision `dad945cac317da394b390f82c7b40691d8a881ed`.
Actual checkpoint geometry: BF16, 20 decoder layers / **19 sparse MoE layers**,
hidden2048, MoE intermediate512, 256 routed experts, top-k8, one shared expert.
This is deliberately not a Qwen3 model or the 100B Flash geometry.
Native sigmoid router / correction bias and normalized selected weights are
unchanged; routed scaling2.5 is applied by the native expert runner.

Existing dInfer upstream `1ffeb961cd258bede74fcf5ca8a416ae6d57b18f` plus the exact pre-existing
bridge patch in `patches/dinfer_existing_bridge.diff`. sglang=0.5.3.post1, deep-ep=1.2.1+73b6ea4, transformers=4.57.0, triton=3.4.0;
PyTorch 2.8.0+cu128, CUDA 12.8, NCCL [2, 27, 3].
Dense TP4 / source-SP4 / physical EP4 / DP1 / PP1, normal high-throughput DeepEP,
eager execution, BF16 Triton FusedMoE. No full-model architecture conversion.

[dInfer](https://github.com/inclusionAI/dInfer) supplies the native mini decoder.
[TEAM](https://arxiv.org/abs/2602.08404) uses temporal/spatial consistency to reduce
expert activation in MoE dLLMs. [Epoch](https://arxiv.org/abs/2609.09748) compiles
block work and routes fresh live/newly-decoded/refresh-required work through EP.
This PoC does not reproduce or benchmark either system. Its optimistic removal
of all prior-accepted output traffic is more favorable than assuming that all
such state can actually be deleted. A large residual does not establish novelty
over TEAM/Epoch; it only motivates a distinct, still-unproven live-work question.

## 2. TRUE physical EP4 validation and GPU mapping

Only mask `2,3,6,7` was used. Physical GPUs2,3,6,7 are H10080GB HBM3;
the saved `nvidia-smi topo -m` has NV18 links among these GPUs. Logical ranks
0,1,2,3 map to physical2,3,6,7 respectively.

| Logical EP rank | Physical GPU | GPU UUID |
| --- | --- | --- |
| 0 | 2 | 24200107-8a7f-de46-1bc8-b81f8d3af13e |
| 1 | 3 | 17488c15-2d4c-5d9e-d503-29b0d959a8a8 |
| 2 | 6 | e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28 |
| 3 | 7 | 4cc26b88-19fc-1988-f9e0-17858aa7a99b |

Each rank owns exactly its contiguous64 experts in all19 sparse layers. Actual
loaded BF16 routed weights are `[64,1024,2048]` and `[64,2048,512]`, totaling
**6,291,456 bytes/expert**. Ownership is not inferred from a synthetic map.
All four ranks call actual DeepEP dispatch and combine, then only their owned
expert weights execute. One B1 request's source rows are partitioned once across
four ranks. Source position uniqueness, actual received row count and received
owner-local path count are asserted for every captured call/layer:
**25,612 physical-buffer validations**.
Output token IDs/NFE match bypass, timing, and trace passes, and all four ranks.
Acceptance-only observer parity additionally passes on the two fixed probes.

Clean-start GPU process audit is retained in `raw/main_v2/clean_start.json`.
Other-GPU workloads are read-only audited and never terminated; shared-fabric
contention from outside the authorized mask remains a timing limitation.

