import os

if os.environ.get("MLLM_COMM_MOTIVATION_ENABLE") == "1":
    from poc_mllm_ep_collective_granularity.instrumentation import install

    install()
    os.environ["MLLM_COMM_MOTIVATION_HOOK_IMPORTED"] = "1"
