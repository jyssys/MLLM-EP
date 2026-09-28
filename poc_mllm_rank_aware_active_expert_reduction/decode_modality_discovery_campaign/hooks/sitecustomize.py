import os

if os.environ.get("MLLM_DMC_TRACE_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.decode_modality_discovery_campaign.rich_trace import install

    install()
elif os.environ.get("MLLM_DMC_PROBE_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.decode_modality_discovery_campaign.structured_probe import install

    install()
