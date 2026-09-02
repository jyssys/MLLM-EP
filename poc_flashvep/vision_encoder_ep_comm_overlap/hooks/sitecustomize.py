import os

if os.environ.get("FLASHVEP_VISION_OVERLAP_DISABLE") != "1":
    from poc_flashvep.vision_encoder_ep_comm_overlap.overlap_hook import install
    install()
