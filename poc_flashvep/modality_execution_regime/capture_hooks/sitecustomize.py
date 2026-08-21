"""Reuse the validated read-only routed-expert capture patch."""

import os

if os.environ.get("FLASHVEP_VISION_TILE_CAPTURE_FIX") == "1":
    from poc_flashvep.vision_tile_motivation.hooks import sitecustomize as _capture_patch
