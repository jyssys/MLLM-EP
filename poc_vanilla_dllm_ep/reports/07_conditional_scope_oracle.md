# Confidence-gated EP-scope oracle

Router mass concentrated on one or two ranks is only a proxy for approximation
risk. It is not an exactness certificate. This report distinguishes measured
rank-mass concentration, an impossible free-scope latency cap, and any actual
verifier-acceptance evidence from the rank-local diagnostic.


Across 416,256 captured refinement tokens, the mean router mass on the best
single physical rank was 45.23%; the best two ranks held 74.21%. Only 0.026% of
tokens had at least 90% mass on one rank, and only 5.45% reached that threshold
with two ranks. Thus local/subgroup scope is not a common high-confidence
execution regime in this model.

The impossible free-all-global-MoE cap is large because the reference EP path
is expensive, but it says nothing about correctness. The measured rank-local
draft shows that physical-scope restriction has poor logit fidelity and no
useful agreement coverage. **Decision: KILL.** A confidence-gated approximate
scope would overlap TEAM/REFLEX/EcoSpec and has no measured quality-safe oracle.
