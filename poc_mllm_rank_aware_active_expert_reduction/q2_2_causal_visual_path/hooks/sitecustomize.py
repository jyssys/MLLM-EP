from __future__ import annotations

import os

if os.environ.get("MLLM_Q22_CAPTURE_ENABLE") == "1":
    from poc_mllm_rank_aware_active_expert_reduction.q2_2_causal_visual_path.causal_vector_capture import install

    install()
    os.environ["MLLM_Q22_CAPTURE_HOOK_IMPORTED"] = "1"
