# Prior boundary and availability audit

This is a kill-test, not a new model, router, load balancer or migration method.
Public endpoints, file inventories and source SHAs are recorded in
`results/model_candidates.json` and `results/prior_source_audit.json`.
These checks are bounded, not an exhaustive claim that no related EP work exists.

- [Sparse Layers are Critical to Scaling Looped Language Models](https://arxiv.org/abs/2605.09165)
  already studies sparse recurrent architectures and their routing behavior.
  Observing loop-dependent expert IDs is not novel here.
- [How to Loop MoE](https://arxiv.org/abs/2609.35751),
  [official code](https://github.com/SR-A-W/how-to-loop-moe) and
  [published weights](https://huggingface.co/ShourenWSR/how-to-loop-moe)
  provide Foil: a shared expert pool and untied per-pass attention. We reuse
  its actual trained native-H16 checkpoint, not a synthetic looped Qwen model.
  The tensor/architecture counts in our audit are verified from the checkpoint.
- [Tying the Loop](https://arxiv.org/abs/2606.16825) and
  [its implementation](https://github.com/epfml/looped-moe) already tie expert
  layers while preserving other layer-specific components. Parameter sharing
  itself is not a systems contribution. Training code is not proof of an
  accessible large pretrained deployment checkpoint.
- [LOOM](https://arxiv.org/abs/2610.01153) and
  [official source](https://github.com/hed-ucas/LOOM) explicitly introduce
  per-loop routers and recurrence-stabilization components. This PoC neither
  introduces those nor modifies a checkpoint to imitate them. At the audited
  revision the README described training/local checkpoints; no published
  download link was identified there. This is not proof of universal absence.
- [Loopie](https://arxiv.org/abs/2607.16051) reports 6B/20B models. Its linked
  preview model endpoints returned HTTP401 and the linked code subpaths
  returned HTTP404 in this audit. Do not substitute a paper's reported model
  for successfully loaded weights, or claim the models cannot exist privately.
- [IFM MoE Middle](https://huggingface.co/IFM/LoopedLM-P1-moe-middle-500b)
  and [xLLM source](https://github.com/ifm-ai/xllm-loop) are a genuine larger
  alternative, with published native BF16 file inventories (~8.78B parameters).
  Our source/config audit finds 25 experts and a native exact-divisibility
  ownership constraint, so native MP/EP4 is not a supported drop-in for this
  artifact. We do not claim these larger weights are absent or loaded here.
- [LLEP](https://proceedings.mlr.press/v306/nguyen26m.html) already addresses
  generic EP imbalance, including movement of expert parameters. Likewise
  [vLLM EP/EPLB](https://github.com/vllm-project/vllm/blob/main/docs/serving/expert_parallel_deployment.md)
  provides generic placement/load-balancing mechanisms. A poor contiguous
  mapping, a balanced histogram, or an oracle redistribution is not novelty.

The only prospective distinction would be a substantial physical critical-path
loss caused specifically by sharing one pool across conflicting loop phases,
remaining beyond a strong global static mapping, with practical net recovery.
We do not establish that distinction here. S1 is a strong multistart **load
proxy** static, not a certified global latency optimum. S2 is future-truth
structural accounting, not measured zero-copy execution or measured speedup.

No migration/replication, new scheduler, speculative loop execution, routing
approximation or training is implemented. The earlier dLLM copy-ROI failure
is motivation for charging movement, not evidence that copy is universally
impossible. Current-device P2P measurements include an intentionally favorable
bulk-coalescing sensitivity bound. Prior live-only 5.77x accounting is not an
Epoch runtime comparison and is not used as a speedup here.
