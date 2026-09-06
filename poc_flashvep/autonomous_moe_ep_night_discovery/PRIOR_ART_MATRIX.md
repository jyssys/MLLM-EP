# Prior-art matrix

No candidate is yet PROMISING.  Literature searches will be attached only
after direct E2E headroom reaches 15%.

| Candidate family | Likely adversarial citations | Required novelty separation |
|---|---|---|
| Cross-DP phase/work partition | ASAP, AEP/AMoE, Layered Prefill, ZeRO-Prefill | online DP assignment/phase sequence must expose a new causal variable beyond length imbalance |
| Continuous-batch composition | PROBE, Gimbal, ExpertPlex, ELDR, Semantic Parallelism | must be EP-critical-path specific, not generic batching |
| Compute/communication dependency | ScMoE, FarSkip-Collective, DeepEP, EPS-MoE | must identify a new high-mass dependency and not generic overlap |
| Host/runtime serialization | vLLM/DeepEP implementation notes | must persist across runtimes or support a systematic principle, not a one-line patch |

