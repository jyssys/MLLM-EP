import os

if os.environ.get("MLLM_RANK_VDS_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.vds_instrumentation import install
    install()

if os.environ.get("MLLM_RANK_STAGE_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.stage_instrumentation import install
    install()
