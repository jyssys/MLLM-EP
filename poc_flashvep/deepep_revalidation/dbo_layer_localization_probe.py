"""Low-volume decoder-layer tensor capture for the DBO numerical gate."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np


_STATE = threading.local()
_INSTALLED = False
_COUNTERS: dict[int, int] = {}
_COUNTER_LOCK = threading.Lock()


def _save(stage: str, value: Any) -> None:
    metadata = getattr(_STATE, "metadata", None)
    if metadata is None:
        return
    if isinstance(value, tuple):
        value = value[0]
    tensor = value.detach()[1].float().cpu().numpy()
    directory = Path(os.environ["FLASHVEP_DBO_LOCALIZATION_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        f"mode-{os.environ['FLASHVEP_DBO_LOCALIZATION_MODE']}"
        f"_dp-{os.environ.get('VLLM_DP_RANK', 'x')}"
        f"_local-{os.environ.get('LOCAL_RANK', 'x')}"
        f"_wave-{metadata['wave']}_step-{metadata['step']}"
        f"_layer-{metadata['layer']:02d}_{stage}.npy"
    )
    np.save(directory / name, tensor)


def install_dbo_layer_localization_probe() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer

    original_init = Qwen3MoeDecoderLayer.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", ""))
        match = re.search(r"layers\.(\d+)$", prefix)
        self._flashvep_layer = int(match.group(1)) if match else -1

        def layer_pre(module: Any, inputs: tuple[Any, ...]) -> None:
            hidden_states, residual = inputs[1], inputs[2]
            if hidden_states.ndim != 2 or hidden_states.shape[0] != 2:
                _STATE.metadata = None
                return
            with _COUNTER_LOCK:
                index = _COUNTERS.get(module._flashvep_layer, 0)
                _COUNTERS[module._flashvep_layer] = index + 1
            _STATE.metadata = {
                "layer": module._flashvep_layer,
                "wave": index // 3,
                "step": index % 3,
            }
            effective = hidden_states if residual is None else hidden_states + residual
            _save("layer_input", effective)

        def attention_post(_module: Any, _inputs: Any, output: Any) -> None:
            _save("attention_output", output)

        def attention_residual_post(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, tuple) and len(output) == 2:
                _save("attention_residual", output[1])

        def moe_pre(_module: Any, inputs: tuple[Any, ...]) -> None:
            _save("dispatch_input", inputs[0])

        def moe_post(_module: Any, _inputs: Any, output: Any) -> None:
            _save("moe_output", output)

        def layer_post(_module: Any, _inputs: Any, output: Any) -> None:
            if isinstance(output, tuple) and len(output) == 2:
                _save("layer_final", output[0] + output[1])
            _STATE.metadata = None

        self.register_forward_pre_hook(layer_pre)
        self.self_attn.register_forward_hook(attention_post)
        self.post_attention_layernorm.register_forward_hook(attention_residual_post)
        self.mlp.register_forward_pre_hook(moe_pre)
        self.mlp.register_forward_hook(moe_post)
        self.register_forward_hook(layer_post)

    Qwen3MoeDecoderLayer.__init__ = patched_init
