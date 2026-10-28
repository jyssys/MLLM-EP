# Prior art and runtime gaps

## Sources

| Source | Verified mechanism | Consequence for this PoC |
|---|---|---|
| [RLT source](https://github.com/InferMatrixLab/vllm-rlt/tree/6b164d3844d35ab8e753df1864f0c87dc180401a) | Separate P/D workers, NIXL writes, chunk streaming, transfer leases, and loop scheduling | These mechanisms are baselines, not new contributions |
| [Splitwise](https://arxiv.org/abs/2311.18677) | Separate prompt and token-generation resources | Phase separation alone does not establish loop-specific novelty |
| [DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin) | Separate phases to reduce interference under latency constraints | Compare goodput and queues, not only isolated transfer time |
| [CDB](https://arxiv.org/abs/2608.09444) | Recurrent scheduling and continuous depth batching | Keep the scheduling quantum distinct from cross-worker ownership |
| [Dory BF16 card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) | Input reinjection, hybrid attention, per-loop and last-loop KV | Cache layouts require separate fidelity tests |

The initial RLT checkout uses `StageEngine`, `StageClient`, and `PDStageConnector`.
It does not contain the older `pd/worker.py` interface.
The PoC therefore uses the current native deployment interface.

## A

A changes compute ownership at a completed recurrent boundary.
The first adapter supports one whole prompt chunk.
The source stops after its assigned loops. The destination executes the remaining loops.
It does not restart the prompt.

The snapshot contains all necessary valid KV, full prompt state, position frontiers, and request RNG state.
Huginn and Dory also require the fixed injection input.
The initial adapter uses a synchronization when it installs destination state.
Its cost belongs in the measured request time.

Generic migration of an unrolled logical-depth stack can reproduce this boundary.
Weight sharing may reduce deployment memory relative to that representation.
It does not remove the causal recurrence dependency.
The experiment must prove an additional benefit before it claims loop-specific novelty.

## B

Upstream `prefill_submitted()` transfers KV after a complete token chunk reaches full depth.
Upstream can overlap that transfer with computation of other chunks.
B instead exposes immutable depth planes at their actual recurrent frontier.
The comparison keeps compute ownership and total depth unchanged.

The initial B implementation uses the same upstream staged G1 path in both arms.
It does not compare staged G1 against atomic G4 and attribute that difference to transfer scheduling.
Layerwise streaming remains a generic competing explanation.

The `shared` layout overwrites its current-token KV between loops.
Intermediate planes in that layout are not immutable final states.
B must not export them as final cache contents.

## C

Upstream initial placement minimizes reserved-block occupancy, then transfer-slot occupancy.
It is not round-robin placement.
C retains the normal full-prefill ownership and handoff policy in its standalone comparison.
Queue, token-count, profiled service-time, and loop-progress controls receive the same legal telemetry.

The first placement smokes used uncalibrated coefficients.
Those results are exploratory. They cannot establish superiority over a strong service-time scheduler.
The repeated main comparison uses frozen coefficients from separate warmed GPU calibration.
`configs/service_calibration.json` records those coefficients and observed batch-size effects.
The model still requires held-out prediction-error analysis before any oracle claim.

## Adapter risks

Upstream rejects Huginn PD and staged boundary-layer prefill explicitly.
The experiment preserves those checks unless an explicit adapter flag enables a qualification run.
The new adapter transfers Huginn's initialization RNG state before destination decode.
It also excludes unused boundary-layer KV planes from the transfer payload.

Dory has 72 alternating attention/MLP sublayers and 36 physical attention caches.
Its model card describes 8 input, 20 recurrent, and 8 output attention/MLP pairs.
These descriptions use different layer-count conventions.
The adapter preserves all 72 sublayers.
