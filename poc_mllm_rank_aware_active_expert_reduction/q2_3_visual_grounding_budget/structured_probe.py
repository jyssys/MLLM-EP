"""Exact semantic K8-to-K6/K4 diagnostic at one frozen decode site.

The retained original weights are renormalized and no replacement expert is
selected. Zero-weight rows can still reach the physical kernel, so this hook
does not demonstrate work removal or latency gain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch

_INSTALLED = False
_CONTEXT = threading.local()


def _control() -> dict[str, Any]:
    path = Path(os.environ["MLLM_VGB_PROBE_CONTROL"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _fingerprints(tensor: torch.Tensor) -> list[str]:
    bits = tensor.detach()[:, :32].contiguous().view(torch.int16).cpu().numpy()
    return [hashlib.sha1(row.tobytes()).hexdigest() for row in bits]


def _layer(prefix: str) -> int:
    match = re.search(r"(?:layers|h)\.(\d+)(?:\.|$)", prefix)
    return int(match.group(1)) if match else -1


def _marker(payload: dict[str, Any]) -> None:
    root = Path(os.environ["MLLM_VGB_PROBE_MARKERS"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{payload['probe_id']}.pid{os.getpid()}.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEKernelModularImpl
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMModel

    original_model = Qwen3MoeLLMModel.forward
    original_init = Qwen3MoeDecoderLayer.__init__
    original_layer = Qwen3MoeDecoderLayer.forward
    original_experts = FusedMoEKernelModularImpl._fused_experts

    def model(self: Any, *args: Any, **kwargs: Any) -> Any:
        _CONTEXT.probe = _control()
        return original_model(self, *args, **kwargs)

    def layer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        prefix = str(kwargs.get("prefix", args[1] if len(args) > 1 else ""))
        self._vgb_layer = _layer(prefix)

    def layer(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = getattr(_CONTEXT, "layer", -1)
        _CONTEXT.layer = int(getattr(self, "_vgb_layer", -1))
        try:
            return original_layer(self, *args, **kwargs)
        finally:
            _CONTEXT.layer = previous

    def experts(self: Any, *args: Any, **kwargs: Any) -> torch.Tensor:
        entry = getattr(_CONTEXT, "probe", {})
        if not entry.get("intervene", False) or int(getattr(_CONTEXT, "layer", -1)) != int(entry.get("target_layer", -2)):
            return original_experts(self, *args, **kwargs)
        names = (
            "in_dtype", "a1q", "a1q_scale", "w1", "w2", "topk_weights", "topk_ids",
            "activation", "global_num_experts", "local_num_experts", "expert_map",
            "apply_router_weight_on_input", "expert_tokens_meta",
        )
        values = dict(zip(names, args, strict=False)); values.update(kwargs)
        x = values["a1q"]
        if x.numel() == 0:
            return original_experts(self, *args, **kwargs)
        matches = [i for i, value in enumerate(_fingerprints(x)) if value == str(entry["target_hidden_hash"])]
        if not matches:
            return original_experts(self, *args, **kwargs)
        if len(matches) != 1:
            raise RuntimeError(f"ambiguous target fingerprint: {matches}")
        row = matches[0]
        ids, weights = values["topk_ids"], values["topk_weights"].clone()
        kept = {int(value) for value in entry["kept_expert_ids"]}
        retained_mass = float(entry["retained_mass"])
        if retained_mass <= 0 or retained_mass > 1.001:
            raise RuntimeError(f"invalid retained mass {retained_mass}")
        owned_kept, owned_removed = 0, 0
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            present = bool(values["topk_weights"][row, slot] > 0)
            if expert in kept:
                weights[row, slot] /= retained_mass
                owned_kept += int(present)
            else:
                weights[row, slot] = 0
                owned_removed += int(present)
        output = original_experts(
            self, values["in_dtype"], x, values.get("a1q_scale"), values["w1"], values["w2"],
            weights, ids, values["activation"], values["global_num_experts"],
            values["local_num_experts"], values.get("expert_map"),
            values["apply_router_weight_on_input"], values.get("expert_tokens_meta"),
        )
        _marker({
            "probe_id": entry["probe_id"], "pid": os.getpid(), "layer": int(entry["target_layer"]),
            "fingerprint": entry["target_hidden_hash"], "keep_k": int(entry["keep_k"]),
            "retained_mass": retained_mass, "owned_kept_count": owned_kept,
            "owned_removed_count": owned_removed,
            "renormalization": "retained_original_weights_sum_to_one",
        })
        return output

    Qwen3MoeLLMModel.forward = model
    Qwen3MoeDecoderLayer.__init__ = layer_init
    Qwen3MoeDecoderLayer.forward = layer
    FusedMoEKernelModularImpl._fused_experts = experts


from vllm import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor


class StructuredProbeLogitsProcessor(AdapterLogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        extra = sampling_params.extra_args or {}
        forced = extra.get("vgb_forced_token_ids")
        if forced is None:
            return
        if not isinstance(forced, list) or not forced:
            raise ValueError("vgb_forced_token_ids must be non-empty")
        step, horizon = int(extra["vgb_probe_step"]), int(extra["vgb_horizon"])
        if step < 0 or step + horizon > len(forced):
            raise ValueError((step, horizon, len(forced)))

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams):
        from vllm.distributed import get_tensor_model_parallel_rank
        extra = params.extra_args or {}
        forced = extra.get("vgb_forced_token_ids")
        if forced is None:
            return None
        forced_ids = tuple(int(value) for value in forced)
        probe_step, horizon = int(extra["vgb_probe_step"]), int(extra["vgb_horizon"])
        root, probe_id = Path(str(extra["vgb_logits_root"])), str(extra["vgb_probe_id"])

        def process(output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
            step = len(output_ids)
            if probe_step <= step < probe_step + horizon and get_tensor_model_parallel_rank() == 0:
                path = root / f"{probe_id}.step{step}.npz"
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(path.suffix + ".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, logits=logits.detach().float().cpu().numpy(),
                                            target_token_id=np.asarray(forced_ids[step], dtype=np.int64),
                                            probe_step=np.asarray(step, dtype=np.int64), probe_id=np.asarray(probe_id))
                    temporary.replace(path)
            if step < len(forced_ids):
                logits.fill_(-float("inf")); logits[forced_ids[step]] = 0.0
            return logits
        return process
