"""Communication-inclusive two-device BF16 decode attention using native FA3."""
from dataclasses import dataclass
import math

from cuda.bindings import runtime as cuda
import torch
import triton
import triton.language as tl
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


def _cuda(operation, *arguments):
    status, *result = operation(*arguments)
    if status != cuda.cudaError_t.cudaSuccess:
        _, reason = cuda.cudaGetErrorString(status)
        raise RuntimeError(f"{operation.__name__}: {status} ({reason!r})")
    return result[0] if len(result) == 1 else tuple(result)


def _ordinal(device):
    index = torch.device(device).index if not isinstance(device, int) else device
    if index is None:
        raise ValueError("Supply an explicit CUDA ordinal")
    return index


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


@dataclass
class KVShard:
    """Prepared post-RoPE KV and valid rows for a causal, one-query decode batch."""

    keys: torch.Tensor
    values: torch.Tensor
    lengths: tuple[int, ...]
    cut: torch.Tensor | None
    block_table: torch.Tensor | None
    used: torch.Tensor | None
    ready: torch.cuda.Event

    @property
    def max_length(self):
        return max(self.lengths)

    @classmethod
    def packed(cls, keys, values, lengths):
        lengths = tuple(lengths)
        cuts = [0]
        for length in lengths:
            cuts.append(cuts[-1] + length)
        if len(keys) != cuts[-1] or keys.shape != values.shape or min(lengths) < 0:
            raise ValueError("Packed K/V requires matching shapes and exact lengths")
        cut = torch.tensor(cuts, dtype=torch.int32, device=keys.device)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(keys.device))
        return cls(keys, values, lengths, cut, None, None, ready)

    @classmethod
    def paged(cls, keys, values, lengths, table, used=None):
        lengths = tuple(lengths)
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("Paged KV expects matching (blocks, block_size, kv_heads, dim) tensors")
        if table.ndim != 2 or table.shape[0] != len(lengths) or table.dtype != torch.int32:
            raise ValueError("Block table must have one int32 row per request")
        if min(lengths) < 0 or max(lengths) > table.shape[-1] * keys.shape[1]:
            raise ValueError("Each declared length must fit its block-table extent")
        if used is None:
            used = torch.tensor(lengths, dtype=torch.int32, device=keys.device)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(keys.device))
        return cls(keys, values, lengths, None, table, used, ready)


def fa3(query, shard, q_starts, scale, *, out=None, lse=False, num_splits=0):
    # Every supplied row is visible to the single latest query. Shards can
    # permute earlier positions because their RoPE transform preceded placement.
    return flash_attn_varlen_func(
        query, shard.keys, shard.values, max_seqlen_q=1, max_seqlen_k=shard.max_length,
        cu_seqlens_q=q_starts, cu_seqlens_k=shard.cut, seqused_k=getattr(shard, "used", None),
        block_table=getattr(shard, "block_table", None), softmax_scale=scale, causal=False,
        out=out, return_softmax_lse=lse, fa_version=3, num_splits=num_splits,
    )


class PeerAccess:
    def __init__(self, decode_device, helper_device):
        for source, target in ((decode_device, helper_device), (helper_device, decode_device)):
            if not torch.cuda.can_device_access_peer(source, target):
                raise RuntimeError(f"CUDA ordinal {source} cannot access {target}")
            with torch.cuda.device(source):
                status, = cuda.cudaDeviceEnablePeerAccess(target, 0)
                if status == cuda.cudaError_t.cudaErrorPeerAccessAlreadyEnabled:
                    cuda.cudaGetLastError()
                elif status != cuda.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"cudaDeviceEnablePeerAccess({source}, {target}) returned {status}")

    @staticmethod
    def copy(destination, source, stream):
        _check_copy(destination, source)
        _cuda(cuda.cudaMemcpyPeerAsync, destination.data_ptr(), destination.device.index,
              source.data_ptr(), source.device.index, _bytes(source), cuda.cudaStream_t(stream.cuda_stream))


def _check_copy(destination, source):
    if (destination.device == source.device or destination.shape != source.shape
            or destination.dtype != source.dtype or not destination.is_contiguous()
            or not source.is_contiguous()):
        raise ValueError("Peer transfer needs contiguous, matching tensors on distinct devices")


def _capture(device, stream, fn):
    """Capture one child on one device, keeping its storage alive."""
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    previous = torch.cuda.current_stream(device)
    with torch.cuda.device(device), torch.cuda.stream(stream):
        if stream != previous:
            stream.wait_stream(previous)
        fn()
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
            output = fn()
    return graph, output


@triton.jit
def _peer_read_query(dst, source_address, n: tl.constexpr, block: tl.constexpr):
    source = source_address.to(tl.pointer_type(tl.bfloat16))
    i = tl.program_id(0) * block + tl.arange(0, block)
    tl.store(dst + i, tl.load(source + i, i < n, 0), i < n)


@triton.jit
def _peer_write_state(answer, stats, answer_address, stats_address,
                      elements: tl.constexpr, lse_elements: tl.constexpr,
                      answer_blocks: tl.constexpr, block: tl.constexpr):
    program = tl.program_id(0)
    if program < answer_blocks:
        answer_dst = answer_address.to(tl.pointer_type(tl.bfloat16))
        i = program * block + tl.arange(0, block)
        tl.store(answer_dst + i, tl.load(answer + i, i < elements, 0), i < elements)
    else:
        stats_dst = stats_address.to(tl.pointer_type(tl.float32))
        i = (program - answer_blocks) * block + tl.arange(0, block)
        tl.store(stats_dst + i, tl.load(stats + i, i < lse_elements, 0), i < lse_elements)


@triton.jit
def _merge_empty_safe(out, d_out, e_out, d_lse, e_lse, empty, batch: tl.constexpr,
                      heads: tl.constexpr, width: tl.constexpr, tile: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    d_valid = tl.load(empty + row) == 0
    e_valid = tl.load(empty + batch + row) == 0
    offset_lse = head * batch + row
    a = tl.where(d_valid, tl.load(d_lse + offset_lse), float("-inf"))
    b = tl.where(e_valid, tl.load(e_lse + offset_lse), float("-inf"))
    m = tl.maximum(a,b)
    d_mass, e_mass = tl.exp(a - m), tl.exp(b - m)
    numerator_heads = (row * heads + head) * width + tl.arange(0,tile)
    valid = tl.arange(0,tile) < width
    x = tl.where(d_valid,tl.load(d_out + numerator_heads,valid,0).to(tl.float32),0.)
    y = tl.where(e_valid,tl.load(e_out + numerator_heads,valid,0).to(tl.float32),0.)
    z = (x * d_mass + y * e_mass) / (d_mass + e_mass)
    tl.store(out + numerator_heads,z,valid)


class GraphPlan:
    """Single-launch composition of GPU-local graph captures and explicit peer copies."""

    def __init__(self, operator, query, local, remote, transport, trace_helper):
        if not remote.max_length:
            raise ValueError("Choose local_graph for entirely local attention")
        self.device = operator.decode_device
        self.keep_alive = (operator, query, local, remote)
        decode, helper = self.device, operator.helper_device
        local_stream = torch.cuda.Stream(device=decode)
        helper_stream = operator.helper_stream
        torch.cuda.synchronize(decode)
        torch.cuda.synchronize(helper)

        local_graph, (local_output, local_lse) = _capture(
            decode, local_stream, lambda: operator.local_part(query, local)
        )
        helper_fn = (lambda: operator.remote_part(remote)) if transport == "memcpy" else (
            lambda: operator.peer_kernel(query,remote))
        self.helper_events = (tuple(torch.cuda.Event(enable_timing=True,external=True) for _ in range(2))
                              if trace_helper else None)

        def run_helper():
            if self.helper_events:
                self.helper_events[0].record(helper_stream)
            output = helper_fn()
            if self.helper_events:
                self.helper_events[1].record(helper_stream)
            return output

        helper_graph, (helper_output, helper_lse) = _capture(helper, helper_stream, run_helper)
        merge_graph, self.output = _capture(
            decode, local_stream, lambda: operator.merge(local_output, local_lse, local, remote)
        )
        self.children = (local_graph, helper_graph, merge_graph)
        self.buffers = (local_output, local_lse, helper_output, helper_lse, operator.empty_tensor)
        self.graph, self.executable = None, None
        try:
            with torch.cuda.device(decode):
                self.graph = _cuda(cuda.cudaGraphCreate, 0)
                local_node = self._child(local_graph, ())
                if transport == "memcpy":
                    query_node = self._copy(operator.remote_query, query, ())
                    helper_node = self._child(helper_graph, (query_node,))
                    answer = self._copy(operator.remote_on_decode, helper_output, (helper_node,))
                    lse = self._copy(operator.remote_lse, helper_lse, (helper_node,))
                    self._child(merge_graph, (local_node, answer, lse))
                else:
                    helper_node = self._child(helper_graph, ())
                    self._child(merge_graph, (local_node, helper_node))
                self.executable = _cuda(cuda.cudaGraphInstantiate, self.graph, 0)
        except RuntimeError:
            self.close()
            raise

    def _copy(self, destination, source, dependencies):
        _check_copy(destination, source)
        return _cuda(cuda.cudaGraphAddMemcpyNode1D, self.graph, list(dependencies), len(dependencies),
                     destination.data_ptr(), source.data_ptr(), _bytes(source),
                     cuda.cudaMemcpyKind.cudaMemcpyDefault)

    def _child(self, graph, dependencies):
        return _cuda(cuda.cudaGraphAddChildGraphNode, self.graph, list(dependencies), len(dependencies),
                     cuda.cudaGraph_t(graph.raw_cuda_graph()))

    def __call__(self):
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            _cuda(cuda.cudaGraphLaunch, self.executable, cuda.cudaStream_t(stream.cuda_stream))
        return self.output

    def close(self):
        with torch.cuda.device(self.device):
            if self.executable is not None:
                _cuda(cuda.cudaGraphExecDestroy, self.executable)
                self.executable = None
            if self.graph is not None:
                _cuda(cuda.cudaGraphDestroy, self.graph)
                self.graph = None


class AssistedAttention:
    """One CUDA queue per device, physically transferred query and partial state."""

    def __init__(self, query_shape, scale=None, decode_device=0, helper_device=1, *, peer=None,
                 num_splits=0):
        self.decode_device = _ordinal(decode_device)
        self.helper_device = _ordinal(helper_device)
        if self.decode_device == self.helper_device:
            raise ValueError("The helper and decode GPUs must be distinct")
        self.decode_home = torch.device("cuda", self.decode_device)
        self.helper_home = torch.device("cuda", self.helper_device)
        self.peer = peer or PeerAccess(self.decode_device, self.helper_device)
        batch, heads, dim = query_shape
        self.scale = scale if scale is not None else 1 / math.sqrt(dim)
        self.num_splits = num_splits
        self.remote_query = torch.empty(query_shape, device=self.helper_home, dtype=torch.bfloat16)
        self.remote_output = torch.empty_like(self.remote_query)
        self.remote_on_decode = torch.empty(query_shape, device=self.decode_home, dtype=torch.bfloat16)
        self.local_output = torch.empty_like(self.remote_on_decode)
        self.result = torch.empty_like(self.remote_on_decode)
        self.remote_lse = torch.empty((heads, batch), device=self.decode_home, dtype=torch.float32)
        self.decode_starts = torch.arange(batch + 1, device=self.decode_home, dtype=torch.int32)
        self.helper_starts = self.decode_starts.to(self.helper_home)
        self.helper_stream = torch.cuda.Stream(device=self.helper_home)
        self.helper_stream.wait_stream(torch.cuda.current_stream(self.helper_home))
        self.query_ready = torch.cuda.Event()
        self.remote_done = torch.cuda.Event()
        self.empty_pattern = None
        self.empty_tensor = None

    @property
    def link_bytes(self):
        return {"query": _bytes(self.remote_query), "partial": _bytes(self.remote_on_decode),
                "lse": _bytes(self.remote_lse)}

    def local(self, query, shard, *, lse=False):
        return fa3(query, shard, self.decode_starts, self.scale, lse=lse, num_splits=self.num_splits)

    def local_part(self, query, shard):
        out, lse = fa3(query, shard, self.decode_starts, self.scale, out=self.local_output, lse=True,
                       num_splits=self.num_splits)
        return out.contiguous(), lse.contiguous()

    def remote_part(self, shard):
        out, lse = fa3(self.remote_query, shard, self.helper_starts, self.scale, out=self.remote_output,
                       lse=True, num_splits=self.num_splits)
        return out.contiguous(), lse.contiguous()

    def peer_kernel(self, query, shard):
        block = 256
        n = query.numel()
        work = triton.cdiv(n, block)
        _peer_read_query[(work,)](self.remote_query,query.data_ptr(),n,block)
        out, lse = self.remote_part(shard)
        k = lse.numel()
        _peer_write_state[(work + triton.cdiv(k,block),)](
            out,lse,self.remote_on_decode.data_ptr(),self.remote_lse.data_ptr(),n,k,work,block
        )
        return out,lse

    def merge(self, local_out, local_lse, local, remote):
        if min(local.lengths) and min(remote.lengths):
            merge_attn_states(self.result, local_out, local_lse, self.remote_on_decode, self.remote_lse)
            return self.result
        # Some FA3 empty intervals leave scratch values uninitialized.
        # Use the known empty pattern to neutralize both LSE and the partial.
        if any(n == m == 0 for n, m in zip(local.lengths, remote.lengths, strict=True)):
            raise ValueError("Every decode row requires a nonempty combined key set")
        pattern = tuple(n == 0 for n in (*local.lengths,*remote.lengths))
        if pattern != self.empty_pattern:
            self.empty_tensor = torch.tensor(pattern,dtype=torch.bool,device=self.decode_home)
            self.empty_pattern = pattern
        batch, heads, width = self.result.shape
        _merge_empty_safe[(batch,heads)](
            self.result,local_out,self.remote_on_decode,local_lse,self.remote_lse,
            self.empty_tensor,batch,heads,width,triton.next_power_of_2(width),
        )
        return self.result

    def split(self, query, local, remote, record=None):
        if not remote.max_length:
            return self.local(query, local)
        decode_stream = torch.cuda.current_stream(self.decode_home)
        self.query_ready.record(decode_stream)
        with torch.cuda.stream(self.helper_stream):
            self.helper_stream.wait_event(self.query_ready)
            if getattr(remote, "ready", None) is not None:
                self.helper_stream.wait_event(remote.ready)
            if record:
                record("remote_begin")
            self.peer.copy(self.remote_query, query, self.helper_stream)
            if record:
                record("query_transferred")
            helper_output, helper_lse = self.remote_part(remote)
            if record:
                record("remote_attention_done")
            self.peer.copy(self.remote_on_decode, helper_output, self.helper_stream)
            self.peer.copy(self.remote_lse, helper_lse, self.helper_stream)
            if record:
                record("remote_result_transferred")
            self.remote_done.record(self.helper_stream)
        if record:
            record("local_begin")
        local_out, local_lse = self.local_part(query, local)
        if record:
            record("local_attention_done")
        decode_stream.wait_event(self.remote_done)
        if record:
            record("merge_begin")
        merged = self.merge(local_out, local_lse, local, remote)
        if record:
            record("merge_done")
        return merged

    def graph(self, query, local, remote, *, transport="memcpy", trace_helper=False):
        if transport not in ("memcpy", "peer_kernel"):
            raise ValueError("Choose memcpy or peer_kernel graph transport")
        return GraphPlan(self, query, local, remote, transport, trace_helper)
