# MLLM-specific increment

There is no held-out incremental co-scheduling value from modality separation under the frozen predictor family:

- S4 minus S3 beneficial union delta: **0.000 experts**, 95% layer-cluster bootstrap CI [0.000, 0.000].
- S4 minus S2 beneficial union delta: **0.000 experts**, 95% CI [0.000, 0.000].
- S4 minus S0 beneficial union delta: **0.689 experts**, 95% CI [0.589, 0.792], but the length control obtains exactly the same held-out waves and metric.

This is stronger than merely observing a nonsignificant timing delta: the frozen held-out S2/S3/S4 partitions are identical. Accordingly, the prior vision-conditioned prediction result does not translate into distinct wave organization on this pool.

Bootstrap clusters are MoE layers (48), not thousands of layer/step events treated as independent requests. The new scheduling held-out is not wholly virgin to the predecessor alpha fit, and it has only eight requests. Nevertheless, the exact S2/S3/S4 equality and the routing-independent oracle's 2.46% ceiling make Stage B unjustified under the specified gate.
