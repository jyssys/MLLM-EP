# Prior-art notes

DeepEP/vLLM already provide asynchronous communication and event dependencies;
this PoC does not claim synchronization itself as novel.  ASAP/Gimbal and
related MoE systems work study broader asynchronous or DP/EP scheduling.  The
narrow candidate would be an online selective wait based on a runtime-visible
outstanding communication state.  The current API exposes neither native
readiness nor a stable event identity, and the diagnostic drain was not robust;
therefore no novelty or method claim is made.
