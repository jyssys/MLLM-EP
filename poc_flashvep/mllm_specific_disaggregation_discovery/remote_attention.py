"""Two-device exact decode attention with measured query and result transfers."""
import ctypes
from pathlib import Path
import sys

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


def bytes_of(tensor):
    return tensor.numel() * tensor.element_size()


class PeerCopies:
    """Enqueue peer copies directly on the overlapping remote stream."""
    def __init__(self):
        cuda = Path(sys.executable).parent.parent / "targets/x86_64-linux/lib64/libcudart.so.12"
        self.library = ctypes.CDLL(str(cuda))
        self.copy = self.library.cudaMemcpyPeerAsync
        self.copy.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                              ctypes.c_size_t, ctypes.c_void_p]
        self.copy.restype = ctypes.c_int
        enable = self.library.cudaDeviceEnablePeerAccess
        enable.argtypes = [ctypes.c_int, ctypes.c_uint]
        enable.restype = ctypes.c_int
        if torch.cuda.device_count() != 2:
            raise ValueError("Assign exactly two leased H200 devices")
        for source, target in ((0, 1), (1, 0)):
            if not torch.cuda.can_device_access_peer(source, target):
                raise RuntimeError("GPU pair does not support direct peer access")
            with torch.cuda.device(source):
                result = enable(target, 0)
                if result not in (0, 704):  # cudaErrorPeerAccessAlreadyEnabled
                    raise RuntimeError(f"Could not enable GPU{source}→GPU{target} peer access, CUDA {result}")
                if result:
                    self.library.cudaGetLastError()

    def __call__(self, destination, source, stream):
        if (destination.device == source.device or destination.shape != source.shape
                or destination.dtype != source.dtype or not destination.is_contiguous()
                or not source.is_contiguous()):
            raise ValueError("Peer copies require different devices and contiguous matching tensors")
        code = self.copy(destination.data_ptr(), destination.device.index,
                         source.data_ptr(), source.device.index, bytes_of(source), stream.cuda_stream)
        if code:
            raise RuntimeError(f"cudaMemcpyPeerAsync rc={code}")


class PackedKV:
    def __init__(self, keys, values, lengths):
        self.keys = keys
        self.values = values
        self.lengths = tuple(lengths)
        self.max_length = max(lengths)
        cuts = [0]
        for size in lengths:
            cuts.append(cuts[-1] + size)
        if cuts[-1] != len(keys) or len(keys) != len(values):
            raise ValueError("Packed KV lengths disagree with key/value rows")
        self.cut = torch.tensor(cuts, device=keys.device, dtype=torch.int32)
        self.ready = torch.cuda.Event()
        self.ready.record(torch.cuda.current_stream(keys.device))


def attention(query, cache, q_starts, scale, *, partial=False):
    """Decode queries are the latest positions: all supplied rows are causal keys."""
    if cache.max_length == 0:
        if not partial:
            raise ValueError("Empty full cache")
        out = torch.zeros_like(query)
        lse = torch.full((query.shape[1], query.shape[0]), -torch.inf,
                         device=query.device, dtype=torch.float32)
        return out, lse
    return flash_attn_varlen_func(
        query, cache.keys, cache.values, max_seqlen_q=1, cu_seqlens_q=q_starts,
        max_seqlen_k=cache.max_length, cu_seqlens_k=cache.cut,
        softmax_scale=scale, causal=False, fa_version=3, return_softmax_lse=partial,
    )


def merge_reference(a, lse_a, b, lse_b):
    """FP32 normalized-output merge with safe all-empty behavior."""
    normalizer = torch.logaddexp(lse_a, lse_b)
    present_a, present_b = lse_a.isfinite(), lse_b.isfinite()
    weight_a = torch.where(present_a, (lse_a - normalizer).exp(), 0).T[..., None]
    weight_b = torch.where(present_b, (lse_b - normalizer).exp(), 0).T[..., None]
    valid_a = torch.where(present_a.T[..., None], a.float(), 0)
    valid_b = torch.where(present_b.T[..., None], b.float(), 0)
    return (valid_a * weight_a + valid_b * weight_b).to(a.dtype)


class DistributedAttention:
    def __init__(self, query_shape, scale, peer):
        self.peer = peer
        self.scale = scale
        batch, heads, _ = query_shape
        self.q_remote = torch.empty(query_shape, dtype=torch.bfloat16, device="cuda:1")
        self.partial = torch.empty(query_shape, dtype=torch.bfloat16, device="cuda:0")
        self.lse = torch.empty((heads, batch), dtype=torch.float32, device="cuda:0")
        self.result = torch.empty_like(self.partial)
        self.q_starts = tuple(torch.arange(batch + 1, dtype=torch.int32, device=f"cuda:{i}") for i in (0, 1))
        self.remote_stream = torch.cuda.Stream(device=1)
        self.remote_stream.wait_stream(torch.cuda.current_stream(1))
        self.q_ready = torch.cuda.Event()
        self.remote_done = torch.cuda.Event()

    @property
    def link_bytes(self):
        return {"query": bytes_of(self.q_remote), "partial": bytes_of(self.partial),
                "lse": bytes_of(self.lse)}

    def local(self, q, local):
        return attention(q, local, self.q_starts[0], self.scale)

    def split(self, q, local, remote, record=None):
        if remote.max_length == 0:
            return self.local(q, local)
        d_stream = torch.cuda.current_stream(0)
        self.q_ready.record(d_stream)
        with torch.cuda.stream(self.remote_stream):
            self.remote_stream.wait_event(self.q_ready)
            self.remote_stream.wait_event(remote.ready)
            if record:
                record("remote_begin")
            self.peer(self.q_remote, q, self.remote_stream)
            if record:
                record("query_transferred")
            part, stat = attention(self.q_remote, remote, self.q_starts[1], self.scale, partial=True)
            # FA3 reports LSE in head-major order. Buffers persist through the consumer wait.
            part, stat = part.contiguous(), stat.contiguous()
            if record:
                record("remote_attention_done")
            self.peer(self.partial, part, self.remote_stream)
            self.peer(self.lse, stat, self.remote_stream)
            if record:
                record("remote_result_transferred")
            self.remote_done.record(self.remote_stream)
        if record:
            record("local_begin")
        local_part, local_lse = attention(q, local, self.q_starts[0], self.scale, partial=True)
        if record:
            record("local_attention_done")
        d_stream.wait_event(self.remote_done)
        if record:
            record("merge_begin")
        if min(local.lengths) and min(remote.lengths):
            merge_attn_states(self.result, local_part.contiguous(), local_lse.contiguous(), self.partial, self.lse)
        else:
            # Empty shards are uncommon with images. Explicit masks handle them even
            # if an empty FA3 interval produces implementation-specific sentinel stats.
            for lengths, stats in ((local.lengths, local_lse), (remote.lengths, self.lse)):
                empty = torch.tensor([not length for length in lengths], device=stats.device)
                stats.masked_fill_(empty.unsqueeze(0), -torch.inf)
            self.result.copy_(merge_reference(local_part, local_lse, self.partial, self.lse))
        if record:
            record("merge_done")
        return self.result
