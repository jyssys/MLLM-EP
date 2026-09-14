#!/usr/bin/env python3
"""Build deterministic EP4 route cases for the RefineStreamEP sweeps."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


WORLD = 4
EXPERTS = 256
TOPK = 8
HIDDEN = 4096


def local_m(global_m: int, rank: int) -> int:
    return global_m // WORLD + int(rank < global_m % WORLD)


def owners_for(kind: str, token: int, rng: random.Random) -> list[int]:
    if kind == "uniform":
        # Exactly two experts/rank and fanout four.
        return [0, 0, 1, 1, 2, 2, 3, 3]
    if kind == "real_like":
        # Broad, mostly remote route support with occasional three-rank fanout.
        anchor = token % WORLD
        if token % 5 == 0:
            seq = [anchor, anchor, (anchor + 1) % 4, (anchor + 1) % 4,
                   (anchor + 2) % 4, (anchor + 2) % 4, anchor, (anchor + 1) % 4]
        else:
            seq = [anchor, anchor, (anchor + 1) % 4, (anchor + 1) % 4,
                   (anchor + 2) % 4, (anchor + 3) % 4, anchor, (anchor + 2) % 4]
        rng.shuffle(seq)
        return seq
    if kind == "mild_skew":
        # Same top-k but a 3/2/2/1 owner split, rotated across tokens.
        a = token % WORLD
        seq = [a] * 3 + [(a + 1) % 4] * 2 + [(a + 2) % 4] * 2 + [(a + 3) % 4]
        rng.shuffle(seq)
        return seq
    if kind == "strong_skew":
        # Concentrated 7/1 split and two-rank fanout.
        a = (token // 16) % WORLD
        seq = [a] * 7 + [(a + 1) % 4]
        rng.shuffle(seq)
        return seq
    raise ValueError(kind)


def route_ids(global_m: int, kind: str, rank: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed + rank * 100003 + global_m * 17)
    result: list[list[int]] = []
    base = sum(local_m(global_m, r) for r in range(rank))
    counters = [[0] * WORLD for _ in range(WORLD)]
    for local_token in range(local_m(global_m, rank)):
        token = base + local_token
        owners = owners_for(kind, token, rng)
        ids = []
        used = set()
        for slot, owner in enumerate(owners):
            # Spread assignments across the 64 experts owned by each rank while
            # preserving unique top-k experts for every token.
            local_e = (token * 13 + slot * 7 + counters[rank][owner]) % 64
            expert = owner * 64 + local_e
            while expert in used:
                local_e = (local_e + 1) % 64
                expert = owner * 64 + local_e
            counters[rank][owner] += 1
            used.add(expert)
            ids.append(expert)
        result.append(ids)
    return result


def summarize(ids_by_rank: list[list[list[int]]]) -> dict:
    owner_load = [0] * WORLD
    fanouts = []
    active = set()
    for source in ids_by_rank:
        for row in source:
            row_owners = {e // 64 for e in row}
            fanouts.append(len(row_owners))
            for expert in row:
                owner_load[expert // 64] += 1
                active.add(expert)
    mean = sum(owner_load) / WORLD if owner_load else 0
    var = sum((x - mean) ** 2 for x in owner_load) / WORLD if mean else 0
    return {
        "rank_load": owner_load,
        "rank_load_cv": math.sqrt(var) / mean if mean else 0.0,
        "mean_fanout": sum(fanouts) / len(fanouts) if fanouts else 0.0,
        "active_experts": len(active),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--m", default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384")
    parser.add_argument("--kinds", default="uniform,real_like,mild_skew,strong_skew")
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ms = [int(x) for x in args.m.split(",") if x]
    kinds = [x for x in args.kinds.split(",") if x]
    with output.open("w") as stream:
        for m in ms:
            for kind in kinds:
                ids = [route_ids(m, kind, rank, args.seed) for rank in range(WORLD)]
                row = {
                    "case_id": f"M{m}_{kind}",
                    "global_m": m,
                    "routing": kind,
                    "hidden": HIDDEN,
                    "experts": EXPERTS,
                    "topk": TOPK,
                    "source_topk_ids": ids,
                    **summarize(ids),
                }
                stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
