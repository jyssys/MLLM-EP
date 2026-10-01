import os

if os.environ.get("MLLM_STATIC4_ENABLE") == "1":
    from poc_mllm_ep_static4_comm_only.instrumentation import install

    install()
    os.environ["MLLM_STATIC4_HOOK_IMPORTED"] = "1"
