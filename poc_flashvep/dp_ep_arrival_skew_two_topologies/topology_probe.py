"""Read-only runtime parallel-group proof for the DP/EP comparison."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

_WRITTEN = False


def _group(group: Any) -> dict[str, Any]:
    return {
        "rank_in_group": int(group.rank_in_group),
        "world_size": int(group.world_size),
        "ranks": [int(x) for x in group.ranks],
    }


def write_once() -> None:
    global _WRITTEN
    if _WRITTEN:
        return
    from vllm.distributed import get_dp_group, get_ep_group, get_pp_group, get_tp_group

    _WRITTEN = True
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_ids = [int(x) for x in visible.split(",") if x.strip().isdigit()]
    local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
    physical = visible_ids[local_rank] if local_rank < len(visible_ids) else None
    from vllm.distributed import get_world_group
    global_rank = int(get_world_group().rank)
    payload = {
        "pid": os.getpid(),
        "physical_gpu": physical,
        "visible_devices": visible,
        "cuda_device": int(torch.cuda.current_device()),
        "global_rank": global_rank,
        "local_rank": local_rank,
        "env_dp_rank": int(os.environ.get("VLLM_DP_RANK", -1)),
        "env_dp_size": int(os.environ.get("VLLM_DP_SIZE", -1)),
        "tp": _group(get_tp_group()),
        "dp": _group(get_dp_group()),
        "ep": _group(get_ep_group()),
        "pp": _group(get_pp_group()),
        "torch_world_size": int(torch.distributed.get_world_size()),
    }
    out = Path(os.environ["FLASHVEP_TOPOLOGY_PROOF_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    rank = payload["global_rank"]
    (out / f"rank{rank}.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
