"""Exact cached implementation of the inherited eight-offer bundle heuristic.

Only the changed layer's offers are recomputed. Selection, ties and load
rejections remain identical to the parent implementation (tested directly).
"""
from collections import Counter
from heapq import merge
from itertools import islice
from poc_dllm_block_local_replica.metrics import members


def bundle_greedy(groups, slots, allowed=None):
    chosen = {layer: 0 for layer in groups}
    remaining = {layer: dict(counter) for layer, counter in groups.items()}
    offers = {}

    def refresh(layer):
        merged = Counter()
        for (_, mask), weight in remaining[layer].items():
            need = mask & ~chosen[layer]
            if need:
                merged[need] += weight
        offers[layer] = sorted(
            ((weight / need.bit_count(), layer, need) for need, weight in merged.items()),
            key=lambda x: (-x[0], x[1], x[2]),
        )

    for layer in groups:
        refresh(layer)
    used = 0
    while used < slots:
        candidates = merge(*offers.values(), key=lambda x: (-x[0], x[1], x[2]))
        shortlist = list(islice((x for x in candidates if x[2].bit_count() <= slots - used), 8))
        best = None
        for _, layer, need in shortlist:
            new = chosen[layer] | need
            gain = sum(weight for (_, mask), weight in remaining[layer].items() if mask & ~new == 0)
            density = gain / need.bit_count()
            if best is None or density > best[0]:
                best = (density, layer, need)
        if best is None:
            break
        _, layer, need = best
        if allowed is not None and not allowed(layer, list(members(need))):
            for key in list(remaining[layer]):
                if key[1] & ~chosen[layer] == need:
                    del remaining[layer][key]
        else:
            chosen[layer] |= need
            used += need.bit_count()
            remaining[layer] = {key: value for key, value in remaining[layer].items() if key[1] & ~chosen[layer]}
        refresh(layer)
    return {layer: set(members(mask)) for layer, mask in chosen.items() if mask}
