# Prior-art matrix

| Work | Main mechanism | Relation to this PoC | Collision assessment |
|---|---|---|---|
| SpecMoE (arXiv:2604.10152) | Self-assisted speculative MoE with affinity and verification | Approximate expert path, but CPU/offload and speculative tokens | Affinity/draft alone is not novel here; downstream EP overlap remains distinct but must be searched |
| Speculative MoE (arXiv:2503.04398) | Speculative token/expert pre-scheduling for communication-efficient MoE | Related prediction/verification objective | Adjacent; not evidence of our downstream hidden-state overlap |
| Speculating Experts (arXiv:2603.19289) | Predict/speculate expert work | Related expert speculation | Adjacent; exact distributed dependency boundary requires comparison |
| ScMoE / FarSkip-Collective | Communication/computation overlap or skipping | Could overlap or reduce MoE work | Generic overlap/skip collision risk |
| Dynamic top-k / expert pruning | Do less expert work | Diagnostic baseline only | Not the intended contribution; matched-quality comparison required |
| Speculative decoding | Draft/verify sequence | Verification/fallback template | Different unit and dependency; use only as systems analogy |

No novelty claim is made from the current bounded data.  A method branch would
require a paper-by-paper search and a current-runtime prototype.
