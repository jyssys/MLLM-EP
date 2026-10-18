# Prior boundary

[Ouro](https://huggingface.co/ByteDance/Ouro-1.4B) supplies the pretrained model and learned gate.
This study does not introduce dynamic depth or train a new gate.

[Continuous Depth Batching](https://github.com/kschwethelm/continuous-depth-batching) supplies the local engine.
Its source performs recurrence, refill, and slot-aware batching.
We reuse those mechanisms. We do not claim them as a contribution.
The public implementation is a research engine, not a production serving system.

[Enabling Dynamic Computation](https://arxiv.org/abs/2610.09013) studies best-available KV retrieval and related training.
We do not reproduce its compact aliases, new training objectives, or model-judge evaluation.
No author repository was verified in this bounded source check.

[vllm-rlt](https://github.com/ThinkFlowLab/vllm-rlt) is an independent implementation.
We do not call it the CDB authors' code.
We did not port it into this experiment.

Generic queue-aware and length-aware DP placement are necessary controls.
A gain against round-robin alone does not establish a depth-specific scheduling opportunity.
The decisive comparison uses the same adaptive model and the strongest non-depth baseline.
A separate true-length oracle isolates output-length information from depth information.

The proposed research question concerns initial request placement across independent CDB replicas.
Any future claim requires additional headroom beyond generic placement.
A correlation between depth and B1 latency is not sufficient.
An offline oracle is not a deployable policy or a measured runtime speedup.
