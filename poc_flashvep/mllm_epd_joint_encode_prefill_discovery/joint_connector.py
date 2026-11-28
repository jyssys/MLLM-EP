"""Stock CUDA IPC for native images and explicit local vision ownership on D."""

import os

from ..mllm_epd_prefill_heavy_token_split.cuda_ipc_ec import CUDAIPCEncoderConnector
from .joint_contract import local_image_prefix


class JointEncoderConnector(CUDAIPCEncoderConnector):
    def has_cache_item(self, identifier):
        if os.environ["MLLM_EPD_ROLE"] == "decode" and identifier.startswith(local_image_prefix()):
            return False
        return super().has_cache_item(identifier)
