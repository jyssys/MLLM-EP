from __future__ import annotations

from poc_mllm_ep_collective_granularity.contract import (
    MODEL, MODEL_REVISION, UPSTREAM_COMMIT, atomic_json, read_json,
    layered_stage_count,
)

GPU_MASK = "0,1,4,5"
LAYERS = 48
SCOPE = "CONTROLLED_NATIVE_LP_EP4_HARNESS"


def partition(groups: int) -> list[tuple[int, int]]:
    if not 1 <= groups <= LAYERS:
        raise ValueError(groups)
    size, extra = divmod(LAYERS, groups)
    edges = [0]
    for i in range(groups):
        edges.append(edges[-1] + size + (i < extra))
    return list(zip(edges[:-1], edges[1:]))


def groups_for(arm: str, prompt_tokens: int) -> int:
    if arm == "LP_AUTO":
        return layered_stage_count(prompt_tokens)
    if arm.startswith("LP_G"):
        return int(arm.removeprefix("LP_G"))
    return 1
