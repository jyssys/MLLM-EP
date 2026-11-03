# Lower-load same-window control

The 128-token, 0.25-second probe configuration passes its clean whole-run Decode screen. Its subsequent qualified P capture fails the central ITL screen. Central ITL p99 is 65.503954 ms. The frozen limit is 65.082876 ms. This failure remains in the final report.

One lower-load capture now tests the existing 0.5-second clean configuration. All other settings, warmup, prompts, arrivals, and SLO limits remain unchanged. The namespace is `qualified_lower_load_joint_ouro128`. The existing exact reference check remains mandatory.

This is one adaptive diagnostic after a failed gate. It is not an independent three-start confirmation or a maximum-capacity estimate. No additional load level will follow within this campaign. A failed gate remains failed, even for a small threshold difference.
