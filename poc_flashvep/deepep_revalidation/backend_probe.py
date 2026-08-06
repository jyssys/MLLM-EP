"""Record the actual vLLM MoE and DBO objects constructed in workers."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


_LOCK = threading.Lock()
_SEEN: set[tuple[str, ...]] = set()
_INSTALLED = False


def _write(kind: str, payload: dict[str, Any]) -> None:
    directory = Path(os.environ["FLASHVEP_DEEPEP_PROOF_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}_pid{os.getpid()}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def install_backend_probe() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    from vllm.model_executor.layers.fused_moe.modular_kernel import (
        FusedMoEKernelModularImpl,
    )
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    original_kernel_init = FusedMoEKernelModularImpl.__init__
    original_ubatch_init = UBatchWrapper.__init__

    def kernel_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_kernel_init(self, *args, **kwargs)
        manager = None
        ep_rank = None
        ep_world = None
        try:
            from vllm.distributed import get_ep_group

            ep = get_ep_group()
            ep_rank = int(ep.rank_in_group)
            ep_world = int(ep.world_size)
            communicator = ep.device_communicator
            manager = getattr(communicator, "all2all_manager", None)
        except Exception:
            pass
        key = (
            type(self.prepare_finalize).__name__,
            type(self.fused_experts).__name__,
            type(manager).__name__ if manager is not None else "None",
        )
        with _LOCK:
            if key in _SEEN:
                return
            _SEEN.add(key)
        _write(
            "moe_backend",
            {
                "prepare_finalize": key[0],
                "expert_backend": key[1],
                "all2all_manager": key[2],
                "ep_rank": ep_rank,
                "ep_world_size": ep_world,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "configured_all2all_backend": os.environ.get(
                    "FLASHVEP_CONFIGURED_ALL2ALL_BACKEND"
                ),
                "configured_dbo": os.environ.get("FLASHVEP_CONFIGURED_DBO"),
            },
        )

    def ubatch_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_ubatch_init(self, *args, **kwargs)
        _write(
            "dbo_wrapper",
            {
                "wrapper": type(self).__name__,
                "num_ubatches": int(
                    self.vllm_config.parallel_config.num_ubatches
                ),
                "enable_dbo": bool(
                    self.vllm_config.parallel_config.enable_dbo
                ),
                "comm_sms": int(getattr(self.sm_control, "comm_sms", -1)),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )

    FusedMoEKernelModularImpl.__init__ = kernel_init
    UBatchWrapper.__init__ = ubatch_init
