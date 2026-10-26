# Prior work and novelty controls

## Sources and scope

The experiment preserves all three pretrained models and their authoritative gates.
It does not revisit expert skipping or expert-output reuse.
The earlier single-GPU negative result remains unchanged.
That result does not establish a distributed EP limit.

| Source | Existing mechanism | Required distinction in this PoC |
| --- | --- | --- |
| [EasyBalance, ICML 2026](https://proceedings.mlr.press/v306/wu26o.html) | Cross-layer workload scheduling for distributed MoE load balance | Cross-pass identical-weight GEMMs need independently ready requests. Generic cross-layer scheduling is not new. |
| [ScMoE, ICML 2025](https://proceedings.mlr.press/v267/cai25c.html) | Shortcut-connected model architecture and communication overlap | This PoC cannot modify pretrained dependencies to create overlap. |
| [FarSkip-Collective, MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/6feb9b30798abcfae937760d183605e1-Abstract-Conference.html) | Changed skip connections and trained model conversion | This PoC preserves the checkpoint and does not distill a converted architecture. |
| [Pre-gated MoE, ISCA 2024](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/05/isca24_pregated_moe_camera_ready.pdf) | A gate selects experts for a later layer | Earlier-pass route overlap cannot supply unknown future hidden states or exact routes. |
| [DeepSeek EPLB](https://github.com/deepseek-ai/EPLB) | Replicate loaded experts and pack replicas across GPUs | Static replicas and ordinary load balancing are strong generic controls, not a recurrent contribution. |
| [Official xllm-loop](https://github.com/ifm-ai/xllm-loop) | Native IFM checkpoint loader and recurrent execution | The thin EP bridge preserves all non-expert execution. |
| [Official Foil implementation](https://github.com/SR-A-W/how-to-loop-moe) | Shared recurrent experts and pass-specific attention | Shared experts do not make pass-specific attention mergeable. |

## Code inspected

The EasyInfra checkout uses commit `6327f7bf24086547de8f622f2a00fd763024b3e9`.
The audit read `easyinfra/modeling_utils/moe.py`.
It also read `easyinfra/generation/blocks/moe/moe_intra_gather.py`.
The latter separates routing, metadata exchange, token exchange, expert compute, and output exchange.
This confirms existing staged compute/communication scheduling in generic MoE infrastructure.
The experiment does not claim that this code supports the three checkpoints unchanged.

The native xLLM checkout uses commit `df0ed1ea415b55f0a944a291f90da9d47c98b5d5`.
The native loader verifies IFM artifact manifests.
The Foil implementation audit uses commit `0f9cc7351658c2437f75379e492f3920f0a0ec19`.
The checkpoint revision is recorded separately from the implementation revision.

The previous physical EP code was also inspected:
`/home/esjung/MLLM-EP-looped-moe-ep/poc_looped_moe_ep/runtime.py`.
Its Foil DeepEP path required a corrected owner-local expert mapping.
Its original dependency stack does not match the current validated PyTorch 2.11 environment.
DeepEP is absent from the current environment.
The new NCCL bridge therefore serves as a research baseline, not a production DeepEP replacement.

## Novelty finding

Shared weights permit identical expert matrices to serve different recurrent passes.
This property does not remove hidden-state dependencies.
B1 has no independent request frontier for exact cross-request coalescing.
At larger batches, only simultaneously ready inputs can support exact merging.

Pass-conditioned placement also needs an incremental information advantage.
Current exact router counts already reveal immediate expert demand.
For a fixed resident set, the tested generic selector uses the same information as the pass-conditioned selector.
The observed small placement differences do not establish a new EP architecture.

No novelty claim survives the tested strong generic controls.
This finding concerns the tested designs and workloads.
It does not exclude other recurrent execution structures.
