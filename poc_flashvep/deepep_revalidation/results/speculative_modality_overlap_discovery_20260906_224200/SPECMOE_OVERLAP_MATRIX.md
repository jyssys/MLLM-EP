# SpecMoE overlap matrix

Primary reference: [SpecMoE, arXiv:2604.10152](https://arxiv.org/abs/2604.10152).
The paper's stated target is self-assisted speculative decoding for
CPU-offloaded MoE, using a draft/target verification path and expert affinity
to reduce expert migration/offload cost.  Similar ideas also appear in
[Speculative MoE, arXiv:2503.04398](https://arxiv.org/abs/2503.04398) and
[Speculating Experts, arXiv:2603.19289](https://arxiv.org/abs/2603.19289).

| SpecMoE concept | Our use | Exact overlap | Important difference | Novelty risk |
|---|---|---|---|---|
| Self-assisted draft execution | Diagnostic surrogate for missing expert terms | Both execute an approximate path before exact verification | We target a provisional hidden state to start downstream distributed compute, not speculative token generation | High if framed as draft MoE alone |
| Expert affinity | Baseline for missing-contribution surrogate | Similar/nearby experts can stand in for unavailable work | Affinity is only an ingredient; the intended contribution is dependency relaxation and critical-path overlap | Very high; not novel by itself |
| Verification | Compare provisional and exact result | Both need target/exact verification | A rejection invalidates nonlinear downstream work; recomputation cost is explicit in our gate | Medium/high |
| Expert migration/offload reduction | Prior work's main systems objective | Both seek to avoid waiting for all expert work | Our runtime is EP/DeepEP with remote experts already resident; issue is late communication completion | Medium |
| Speculative tokens | SpecMoE output unit | Both may speculate before exact result | Our unit is a hidden-state/downstream computation boundary, not new output tokens | Medium |
| Distributed EP overlap window | Not the SpecMoE claim | No exact match identified in the reviewed abstract | We ask whether approximate partial combine can overlap downstream compute with late remote experts | Needs adversarial literature review before any method claim |
