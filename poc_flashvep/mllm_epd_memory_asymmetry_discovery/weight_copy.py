"""Physically copy the original Qwen language tensors from E or pinned host staging."""

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

from cuda.bindings import runtime as cuda
from safetensors import safe_open
import torch

from ..encoder_assisted_decode_attention.peer_attention import PeerAccess, _cuda


def checkpoint_language(snapshot):
    weight_index = snapshot / "model.safetensors.index.json"
    index_data = weight_index.read_bytes()
    keys_by_shard = defaultdict(list)
    for name, shard in json.loads(index_data)["weight_map"].items():
        if name.startswith(("model.", "lm_head.")):
            keys_by_shard[shard].append(name)
    weights = []
    for shard_name, keys in keys_by_shard.items():
        with safe_open(snapshot / shard_name, framework="pt", device="cpu") as shard:
            for name in keys:
                parameter = shard.get_tensor(name)
                if parameter.dtype != torch.bfloat16:
                    raise ValueError(f"Expected BF16 model tensor {name}")
                weights.append(parameter.contiguous().view(torch.uint8).flatten())
    if not weights or not any(shard.startswith("model-") for shard in keys_by_shard):
        raise ValueError("Expected original sharded Qwen language weights")
    return weights, hashlib.sha256(index_data).hexdigest()


def transfer_host(weights, destination, stream, buffers):
    available = [torch.cuda.Event() for _ in buffers]
    busy = [False] * len(buffers)
    tile_bytes = buffers[0].numel()
    current = 0
    start = time.perf_counter()
    for weight, device_weight in zip(weights, destination, strict=True):
        for offset in range(0, weight.numel(), tile_bytes):
            nbytes = min(tile_bytes, weight.numel() - offset)
            if busy[current]:
                available[current].synchronize()
            staging = buffers[current][:nbytes]
            staging.copy_(weight[offset:offset + nbytes])
            device_weight[offset:offset + nbytes].copy_(staging, non_blocking=True)
            available[current].record(stream)
            busy[current] = True
            current = (current + 1) % len(buffers)
    stream.synchronize()
    return 1e3 * (time.perf_counter() - start)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 0 < args.destination < torch.cuda.device_count() or args.repeats < 1:
        parser.error("Select distinct visible CUDA ordinals and positive repeats")
    if any(torch.cuda.get_device_name(i) != "NVIDIA H200" for i in (0, args.destination)):
        parser.error("Use the H200 campaign GPU lease")
    original, index_hash = checkpoint_language(args.model)
    total = sum(t.numel() for t in original)
    for device in (0, args.destination):
        if torch.cuda.mem_get_info(device)[0] < total + 4 * 2**30:
            raise MemoryError("Full tensor allocation would consume the 4 GiB guard")

    peer = PeerAccess(args.destination, 0)
    e_resident = [weight.to("cuda:0") for weight in original]
    device = torch.device(f"cuda:{args.destination}")
    scratch = [torch.empty(weight.shape, dtype=torch.uint8, device=device) for weight in original]
    buffers = [torch.empty(64 * 2**20, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
    stream = torch.cuda.Stream(device=device)
    torch.cuda.synchronize(0)
    torch.cuda.synchronize(device)
    gpu_times, device_wall_times, cpu_times = [], [], []
    with torch.cuda.device(device), torch.cuda.stream(stream):
        for _ in range(args.repeats):
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start = time.perf_counter()
            begin.record()
            for src, dst in zip(e_resident, scratch, strict=True):
                peer.copy(dst, src, stream)
            end.record()
            end.synchronize()
            device_wall_times.append(1e3 * (time.perf_counter() - start))
            gpu_times.append(begin.elapsed_time(end))
            cpu_times.append(transfer_host(original, scratch, stream, buffers))
    checked = sorted({0, len(original) // 2, len(original) - 1})
    if any(not torch.equal(original[i], scratch[i].cpu()) for i in checked):
        raise RuntimeError("Copied checkpoint scratch tensors do not match the host")
    result = {"evidence": "MEASURED_PHYSICAL", "model": str(args.model.resolve()),
              "weight_index_sha256": index_hash, "tensors": len(original), "actual_language_weight_bytes": total,
              "E_to_destination_gpu_event_ms": gpu_times,"E_to_destination_wall_ms": device_wall_times,
              "host_staged_to_destination_wall_ms": cpu_times,
              "E_to_destination_wall_median_ms": median(device_wall_times),
              "host_staged_to_destination_wall_median_ms": median(cpu_times),
              "host_staging_buffers": [slot.numel() for slot in buffers], "sample_tensor_checks": len(checked),
              "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "scope": "Copy into scratch BF16 checkpoint storage. Matched host wall ends at destination CUDA completion. "
                       "Host staging includes host memcpy. No runtime model switch, "
                       "vision interference, loading of new graphs or online multimodel requests."}
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
