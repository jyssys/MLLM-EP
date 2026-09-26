"""Install read-only Part-1 profilers inside spawned vLLM workers."""

import os

if os.environ.get("MLLM_EP_FOUNDATION_PROFILE") == "1":
    from poc_flashvep.live_traffic_matrix_validation.instrumentation import install as install_ep
    from poc_mllm_ep_foundation.profiling import install as install_full

    install_ep()
    install_full()
elif os.environ.get("MLLM_EP_FOUNDATION_PROFILE_EP") == "1":
    from poc_flashvep.live_traffic_matrix_validation.instrumentation import install as install_ep

    install_ep()
elif os.environ.get("MLLM_EP_FOUNDATION_PROFILE_FULL") == "1":
    from poc_mllm_ep_foundation.profiling import install as install_full

    install_full()
