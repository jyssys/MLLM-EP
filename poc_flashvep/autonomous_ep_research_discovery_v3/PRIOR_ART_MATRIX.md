# Prior-art adversarial screen

This sprint did not promote a paper finalist, so no exact novelty claim is
made. The table records why the current observations are not yet blue-ocean.

| Candidate | Closest prior work / runtime | What overlaps | Remaining gap needed for novelty | Decision |
|---|---|---|---|---|
| fixed-shape async tail | DeepEP `previous_event`, vLLM DeepEP HT, ASAP/Gimbal-style overlap | asynchronous communication and synchronization affect latency | request-level high-mass effect plus a nontrivial debt-aware method | closed by prior direct E2E cap |
| concurrency/phase geometry | continuous batching and layered/chunked prefill systems | batch composition changes GPU critical path | a shape variable orthogonal to token count with large E2E oracle | probe only; not exact collision yet |
| expert-vs-dispatch phase mix | MoE kernel tuning, DA-MoE/TEMPO | expert histogram/makespan and kernel cost | distributed runtime interaction after load controls | no causal evidence |
| modality phase mix | SpaceServe/RESONATOR | MLLM encoder/LLM resource sharing | modality × MoE execution with direct E2E gain | current signal includes encoder cost |
| communication-SM knob | DeepEP/vLLM documented configuration | static resource tuning | systematic oracle gap beyond existing knob | previous lower envelope <=2.1% |
| fanout geometry | DA-MoE/TEMPO-like routing representations | route distribution affects execution | incremental incidence signal after load controls | measured null, closed |

Search status: no candidate met the promotion gate (`>=15%` direct E2E perfect
oracle plus controlled signal), therefore this sprint intentionally avoids a
strong novelty claim.

Adversarial reading included the published/primary descriptions of [AEP/AMoE](https://arxiv.org/abs/2505.08944)
(asynchronous expert parallelism and defragmenting), [EPS-MoE](https://arxiv.org/abs/2410.12247)
(load-dependent kernel/communication scheduling), [Semantic Parallelism](https://proceedings.iclr.cc/paper_files/paper/2026/hash/f0552f14388d95b19740dee809f5cad1-Abstract-Conference.html)
(model-data co-scheduling), and [Layered Prefill](https://proceedings.mlsys.org/paper_files/paper/2026/hash/c0f460c6d63599ea870ba9db63dc96a9-Abstract-Conference.html)
(prefill/decode stage composition). These make a generic batching or overlap
effect high-risk for novelty; a future candidate must isolate a variable not
already captured by those systems.
