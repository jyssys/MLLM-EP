from __future__ import annotations

import os

if os.environ.get("MLLM_DAMAGE_PROBE_ENABLE") == "1":
    from poc_mllm_vision_conditioned_decode_damage_skipping.functional_probe import install

    install()
    os.environ["MLLM_DAMAGE_PROBE_HOOK_IMPORTED"] = "1"
