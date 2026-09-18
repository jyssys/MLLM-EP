# DeltaEP quality

The bitmap+literals representation reconstructs every BF16 word by exact XOR inversion; unit tests cover unchanged, partially changed, and random tensors. Therefore semantic quality is unchanged by construction. No lossy path or quality rollout was run. The captured combine vector is the post-routed-MoE combined-output predictor; destination-local branch-vector compression remains unmeasured and is not overclaimed.
