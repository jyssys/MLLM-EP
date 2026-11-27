"""Page-backed prompt state and append-only generated state for staged serving."""
from dataclasses import dataclass
import math

import torch


PAGE = 128
GIB = 2**30


class PageIds:
    def __init__(self, count):
        self.free = list(reversed(range(count)))
        self.owners = set()
        self.count = count
        self.peak_used = 0

    def fits(self, rows):
        return math.ceil(rows / PAGE) <= len(self.free)

    def acquire(self, rows):
        pages = math.ceil(rows / PAGE)
        if not self.fits(rows):
            raise ValueError(f"Page exhaustion for {rows} rows")
        chosen = [self.free.pop() for _ in range(pages)]
        self.owners.update(chosen)
        self.peak_used = max(self.peak_used, len(self.owners))
        return chosen

    def release(self, pages):
        if len(pages) != len(set(pages)) or not self.owners.issuperset(pages):
            raise ValueError("Pages released twice or by the wrong owner")
        self.owners.difference_update(pages)
        self.free.extend(pages)


class Arena(PageIds):
    def __init__(self, dev, layers, heads, dim, gib):
        self.device = torch.device(dev)
        self.bytes_per_row = layers * 2 * heads * dim * torch.bfloat16.itemsize
        count = int(gib * GIB) // (PAGE * self.bytes_per_row)
        super().__init__(count)
        if not count:
            raise ValueError("Budget is smaller than one physical KV block")
        with torch.cuda.device(self.device):
            self.cache = torch.empty((layers, 2, count, PAGE, heads, dim),
                                     device=self.device, dtype=torch.bfloat16)
            self.ingress = torch.cuda.Stream(device=self.device)

    def enqueue(self, prefix, indices, pages):
        if indices.numel() == 0:
            return None
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # These controls are admission-time work, independent of the per-layer decode path.
        with torch.cuda.stream(self.ingress):
            # For P-retained KV, index selection and index allocation share this stream.
            indices = indices.to(prefix.layers[0].keys.device)
            begin.record()
            slots = torch.tensor(pages, device=self.device)[:, None] * PAGE
            slots = (slots + torch.arange(PAGE, device=self.device)).flatten()[:indices.numel()]
            for index, layer in enumerate(prefix.layers):
                for direction, values in enumerate((layer.keys, layer.values)):
                    values = values[0].transpose(0, 1)
                    source = values[indices].contiguous().to(self.device, non_blocking=True)
                    self.cache[index, direction].flatten(0, 1).index_copy_(0, slots, source)
            end.record()
        return begin, end

    @staticmethod
    def wait(transfer):
        if transfer is None:
            return 0.
        begin, end = transfer
        end.synchronize()
        return begin.elapsed_time(end)

    def receipt(self):
        return {"device": str(self.device), "allocated_pages": self.count, "peak_active_pages": self.peak_used,
                "bytes_per_row": self.bytes_per_row, "allocated_bytes": self.count * PAGE * self.bytes_per_row}


@dataclass
class Placement:
    d_pages: list
    helper_pages: list
    d_rows: int
    helper_rows: int
    source_rank: int
    method: str


def partition(length, visual, mode, ratio):
    all_rows = torch.arange(length)
    if mode == "local":
        return all_rows, torch.empty(0, dtype=torch.long)
    if mode == "visual_all":
        selected = visual
    elif mode == "visual_first":
        count = min(length, int(length * ratio))
        selected = visual[:count]
        if selected.numel() < count:
            available = torch.ones(length, dtype=torch.bool)
            available[selected] = False
            selected = torch.cat((selected, all_rows[available][:count-selected.numel()])).sort().values
    elif mode == "matched_visual_count":
        selected = all_rows[:visual.numel()]
    elif mode == "position":
        selected = all_rows[:int(length * ratio)]
    else:
        raise ValueError(f"Unsupported static partition {mode}")
    remaining = torch.ones(length, dtype=torch.bool)
    remaining[selected] = False
    return all_rows[remaining], selected


def page_metadata(arena, pages, lengths):
    width = max(1, max(map(len, pages)))
    table = torch.tensor([row + [0]*(width-len(row)) for row in pages],
                         dtype=torch.int32, device=arena.device)
    used = torch.tensor(lengths, dtype=torch.int32, device=arena.device)
    return table, used


class DecodePages:
    def __init__(self, d, helper, requests, kernel, heads, dim):
        self.d, self.helper = d, helper
        self.kernel, self.heads, self.dim = kernel, heads, dim
        self.graph_batch = None
        p = [r.placement for r in requests]
        # First sampled token emitted by P is consumed into the first D slot.
        gen = [len(r.tokens) for r in requests]
        self.d_lengths = [m.d_rows + produced for m, produced in zip(p, gen, strict=True)]
        self.h_lengths = [m.helper_rows for m in p]
        self.d_table, self.d_used = page_metadata(d, [m.d_pages for m in p], self.d_lengths)
        self.q_starts = torch.arange(len(p) + 1, dtype=torch.int32, device=d.device)
        self.slots = torch.tensor([
            m.d_pages[(length-1) // PAGE]*PAGE + (length-1) % PAGE
            for m, length in zip(p, self.d_lengths, strict=True)
        ], device=d.device)
        self.h_table, self.h_used = (page_metadata(helper, [m.helper_pages for m in p], self.h_lengths)
                                     if any(self.h_lengths) else (None, None))
        if self.h_used is not None:
            # Record on the metadata allocation stream before the other helper stream consumes it.
            self.h_ready = torch.cuda.Event()
            self.h_ready.record(torch.cuda.current_stream(helper.device))

    def append(self, layer, keys, values):
        self.d.cache[layer, 0].flatten(0, 1).index_copy_(0, self.slots, keys)
        self.d.cache[layer, 1].flatten(0, 1).index_copy_(0, self.slots, values)

    def attend(self, layer, query):
        if self.graph_batch is not None:
            return self.graph_batch.attend(layer, query)
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        if self.h_used is None:
            return flash_attn_varlen_func(
                query, self.d.cache[layer, 0], self.d.cache[layer, 1],
                max_seqlen_q=1, cu_seqlens_q=self.q_starts, seqused_k=self.d_used,
                block_table=self.d_table, max_seqlen_k=max(self.d_lengths),
                softmax_scale=self.dim ** -.5, causal=False, fa_version=3,
            )
        from .peer_attention import KVShard
        local = KVShard(self.d.cache[layer, 0], self.d.cache[layer, 1], self.d_lengths,
                        None, self.d_table, self.d_used, None)
        remote = KVShard(self.helper.cache[layer, 0], self.helper.cache[layer, 1], self.h_lengths,
                         None, self.h_table, self.h_used, self.h_ready)
        return self.kernel.split(query, local, remote)
