"""vLLM 0.26 exact KV offload with a second GPU as passive storage.

The regular vLLM scheduler owns all hashes, page reservations, reference counts,
and eviction decisions. Only the worker's storage device changes. No decoder
kernels run on the peer, and E takes no part in an admitted decode step.
"""

from collections import deque
import json
import os
from pathlib import Path

import torch
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.gpu_worker import (
    CPUOffloadingWorker,
    SingleDirectionOffloadingHandler,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

from poc_flashvep.encoder_assisted_decode_attention.peer_attention import PeerAccess

logger = init_logger(__name__)


def swap_peer_blocks(src, dst, sizes, *, is_src_access_order_any):
    # Keep the read of peer VRAM ordered; the host-only fast path's flag is
    # deliberately unused. vLLM reports transfer completion by CUDA event.
    ops.swap_blocks_batch(src, dst, sizes, is_src_access_order_any=False)


class PeerDirection(SingleDirectionOffloadingHandler):
    def __init__(self, local, remote, blocks_per_chunk, group_refs, store):
        if not local or len(local) != len(remote):
            raise ValueError("Local and peer KV page groups must match")
        for source, target in zip(local, remote, strict=True):
            if (not source.is_cuda or not target.is_cuda or source.device == target.device
                    or source.ndim != 2 or target.ndim != 2 or source.dtype != torch.int8
                    or target.dtype != torch.int8 or target.shape[1] != source.shape[1] * blocks_per_chunk):
                raise ValueError("Peer KV storage must contain page-aligned byte tensors on a second CUDA device")

        self.src_tensors = local if store else remote
        self.dst_tensors = remote if store else local
        self.gpu_to_cpu = store  # The parent uses this to wait for the producer stream.
        self.kv_cache_groups_data_refs = group_refs
        self._swap_blocks_batch = swap_peer_blocks
        self.src_blocks_per_chunk = 1 if store else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if store else 1
        self._mmap_region = None
        self._transfer_events = {}
        self._transfers = deque()
        self._stream_pool = []
        self._event_pool = []
        self._buffer_pool = []


class PeerOffloadingWorker(CPUOffloadingWorker):
    def __init__(self, kv_caches, blocks_per_chunk, pages, tier_device, guard_bytes):
        local_device = torch.cuda.current_device()
        if pages <= 0 or tier_device == local_device or not 0 <= tier_device < torch.cuda.device_count():
            raise ValueError("Physical peer GPU and positive KV page capacity required")
        PeerAccess(local_device, tier_device)
        torch.cuda.set_device(local_device)
        local = [part.tensor.view(torch.int8).view(-1, part.page_size_bytes) for part in kv_caches.tensors]
        if any(tensor.device.index != local_device for tensor in local):
            raise ValueError("All canonical pages must be on the active prefill GPU")
        capacity = sum(tensor.shape[1] for tensor in local) * blocks_per_chunk * pages
        before, _ = torch.cuda.mem_get_info(tier_device)
        if capacity + guard_bytes > before:
            raise RuntimeError(f"Peer KV capacity {capacity} and guard {guard_bytes} exceed free VRAM {before}")
        with torch.cuda.device(tier_device):
            remote = [torch.zeros((pages, tensor.shape[1] * blocks_per_chunk), dtype=torch.int8,
                                  device=tier_device) for tensor in local]
            torch.cuda.synchronize(tier_device)
            after, _ = torch.cuda.mem_get_info(tier_device)
        if after < guard_bytes:
            raise RuntimeError(f"Actual peer allocation leaves {after} B, below the {guard_bytes} B guard")
        logger.info("Passive peer KV storage local=%d peer=%d pages=%d allocated_bytes=%d "
                    "device_free_before=%d device_free_after=%d guard_bytes=%d",
                    local_device, tier_device, pages, capacity, before, after, guard_bytes)
        session_root = os.environ.get("MLLM_EPD_SESSION_ROOT")
        if session_root:
            with (Path(session_root) / "state" / "peer_storage.json").open("x") as stream:
                json.dump({"active_cuda_device": local_device, "peer_cuda_device": tier_device,
                           "capacity_bytes": capacity, "pages": pages, "blocks_per_chunk": blocks_per_chunk,
                           "peer_free_before_bytes": before, "peer_free_after_bytes": after,
                           "peer_guard_bytes": guard_bytes,
                           "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, stream, indent=2)
                stream.write("\n")
        self._store_handler = PeerDirection(local, remote, blocks_per_chunk, kv_caches.group_data_refs, True)
        self._load_handler = PeerDirection(local, remote, blocks_per_chunk, kv_caches.group_data_refs, False)


class PeerOffloadingSpec(CPUOffloadingSpec):
    def create_worker(self, kv_caches):
        return PeerOffloadingWorker(kv_caches, self.blocks_per_chunk, self.num_blocks,
                                    int(self.extra_config["tier_device"]),
                                    int(self.extra_config["tier_guard_bytes"]))
