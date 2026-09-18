# FrontierEP AR control

A normal causal AR KV cache already removes every prior generated token from fresh token execution. There is no repeatedly refined within-block committed frontier, so incremental one-/two-finalization sealing beyond standard AR caching is **0% by construction**. The structural opportunity is dLLM-specific, but the tested dLLM semantic change is unsafe.
