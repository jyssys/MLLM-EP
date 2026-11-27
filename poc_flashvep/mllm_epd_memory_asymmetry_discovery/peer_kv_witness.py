"""Exercise the actual vLLM worker batch-DMA path, including scatter and offset loads."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalKVCacheTensor, CanonicalKVCaches, GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker

from .peer_kv import PeerOffloadingWorker


def check(backend, factor):
    # 28 layers, Qwen2.5-VL-7B BF16 K+V, Block128. Different physical ids
    # make an identity or contiguous-copy implementation fail the witness.
    page = 128 * 2 * 4 * 128 * 2
    torch.cuda.set_device(0)
    torch.manual_seed(2026112802)
    tensors = [torch.randint(-128, 127, (30, page), dtype=torch.int8, device=0) for _ in range(28)]
    caches = CanonicalKVCaches([CanonicalKVCacheTensor(tensor, page) for tensor in tensors],
                               [[CanonicalKVCacheRef(index, page) for index in range(28)]])
    if backend == "peer":
        worker = PeerOffloadingWorker(caches, factor, 8, tier_device=1, guard_bytes=0)
    else:
        worker = CPUOffloadingWorker(caches, factor, 8)
    try:
        stored = [6, 10, 3, 0, 8, 15][:3 * factor]
        destinations = [22, 27, 23, 29, 24, 26][:3 * factor]
        offset = factor - 1
        destinations = destinations[offset:]
        original = stored[offset:]
        source = GPULoadStoreSpec(stored, [len(stored)], [0])
        target = GPULoadStoreSpec(destinations, [len(destinations)], [offset])
        remote = CPULoadStoreSpec([7, 2, 5])
        expected = [tensor[original].clone() for tensor in tensors]
        untouched = [tensor[21].clone() for tensor in tensors]
        if not worker.submit_store(1, source, remote):
            raise RuntimeError("Store declined")
        worker.wait({1})
        if not worker.submit_load(2, remote, target):
            raise RuntimeError("Load declined")
        worker.wait({2})
        results = worker.get_finished()
        payload_equal = [bool(torch.equal(tensor[destinations], result)) for tensor, result in zip(tensors, expected, strict=True)]
        guards_equal = [bool(torch.equal(tensor[21], result)) for tensor, result in zip(tensors, untouched, strict=True)]
        return {"backend": backend, "page_size_bytes": page, "blocks_per_chunk": factor,
                "source_block_ids": stored, "target_block_ids": destinations, "offload_block_ids": remote.block_ids.tolist(),
                "partial_first_page_skip": offset, "all_layer_KV_byte_equal": payload_equal,
                "untouched_guard_equal": guards_equal, "transfers": [asdict(result) for result in results],
                "passed": len(results) == 2 and all(result.success for result in results)
                and all(payload_equal) and all(guards_equal)}
    finally:
        worker.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = [check(backend, factor) for backend in ("peer", "cpu") for factor in (1, 2)]
    with args.out.open("x") as stream:
        json.dump({"device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                   "evidence": "CORRECTNESS_ONLY", "results": results}, stream, indent=2)
        stream.write("\n")
    if not all(result["passed"] for result in results):
        raise RuntimeError("A byte-level round trip failed")


if __name__ == "__main__":
    main()
