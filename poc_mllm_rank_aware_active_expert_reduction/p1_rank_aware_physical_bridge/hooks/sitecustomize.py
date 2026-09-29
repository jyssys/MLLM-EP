import os

if os.environ.get("MLLM_P1_BRIDGE_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.p1_rank_aware_physical_bridge.instrumentation import install

    install()
    os.environ["MLLM_P1_BRIDGE_HOOK_IMPORTED"] = "1"
