import os

if os.environ.get("MLLM_VGB_CAPTURE_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.subspace_capture import install

    install()
    os.environ["MLLM_VGB_CAPTURE_HOOK_IMPORTED"] = "1"
elif os.environ.get("MLLM_VGB_PROBE_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.q2_3_visual_grounding_budget.structured_probe import install

    install()
    os.environ["MLLM_VGB_PROBE_HOOK_IMPORTED"] = "1"
