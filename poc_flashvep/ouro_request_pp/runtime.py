"""Four independent CUDA workers. FIFO per rank. Whole-request stages only.

This bounded implementation uses one CPU thread and CUDA stream per physical GPU.
Cross-rank activation copies use CUDA peer copy, not CPU tensor serialization.
No global barrier occurs inside a stage, loop, or generation step.
"""
from dataclasses import dataclass
import queue
import threading
import time
import traceback
import torch
from .contract import route
from .model_adapter import Part
from .peer_pool import PeerPool


@dataclass
class Item:
    rid: str
    step: int
    position: int
    visit: int
    payload: torch.Tensor
    submitted: float


class Pipeline:
    def __init__(self, placement, preallocated_recv=False, precompute_rotary=False):
        self.placement = placement
        self.peer_pools=[PeerPool(rank) for rank in range(4)] if preallocated_recv else None
        self.parts = [Part(rank, placement) for rank in range(4)]
        for part in self.parts:part.precompute_rotary=precompute_rotary
        self.ready = [queue.Queue() for _ in range(4)]
        self.completed = queue.Queue()
        self.threads = [threading.Thread(target=self.worker, args=(rank,), daemon=True) for rank in range(4)]
        self.path = None
        self.events = []
        self.lock = threading.Lock()
        for thread in self.threads:
            thread.start()

    @torch.inference_mode()
    def worker(self, rank):
        torch.cuda.set_device(rank)
        stream = torch.cuda.Stream(device=rank)
        with torch.cuda.stream(stream):
            while True:
                item = self.ready[rank].get()
                if item is None:
                    return
                try:
                    self.execute(rank, stream, item)
                except BaseException:
                    self.completed.put(('error', traceback.format_exc()))
                    return

    def execute(self, rank, stream, item):
        part = self.parts[rank]
        _, loop, first, stop = self.path[item.visit]
        host_start = time.perf_counter()
        label = f'{self.placement}|{item.rid}|step{item.step}|R{loop+1}|L{first}-{stop}'
        torch.cuda.nvtx.range_push(label)
        if self.trace:
            a, b, c, d = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            a.record(stream)
        else:
            d = torch.cuda.Event(enable_timing=False)
        payload = item.payload
        source = payload.device.index if payload.is_cuda else -1
        transfer_bytes = 0
        transfer_kind = None
        if payload.device != part.device:
            transfer_kind = 'hidden' if payload.dtype == torch.bfloat16 else ('token_return' if payload.is_cuda else 'input_H2D')
            transfer_bytes = payload.numel() * payload.element_size()
            if self.trace:torch.cuda.nvtx.range_push(f'P2P|{transfer_kind}|{source}>{rank}')
            if self.peer_pools is not None and payload.is_cuda:
                payload = self.peer_pools[rank].transfer(item.rid,payload)
            else:
                payload = payload.to(part.device, non_blocking=True)
            if self.trace:torch.cuda.nvtx.range_pop()
        if self.trace:
            b.record(stream)
        hidden = part.embedding(payload) if item.visit == 0 else payload
        if self.trace:torch.cuda.nvtx.range_push('CORE')
        hidden = part.stage(hidden, item.rid, item.step, item.position, loop, first, stop)
        if self.trace:torch.cuda.nvtx.range_pop()
        if self.trace:
            c.record(stream)
        terminal = item.visit == len(self.path) - 1
        logits = token = None
        if terminal:
            if self.trace:torch.cuda.nvtx.range_push('HEAD')
            logits = part.head(hidden[:, -1:, :])
            token = logits[:, -1].argmax(-1).view(1, 1)
            if self.trace:torch.cuda.nvtx.range_pop()
        d.record(stream)
        # This wait publishes this stage only. Other ranks continue independently.
        d.synchronize()
        host_end = time.perf_counter()
        self.queue_wait_ms[rank] += (host_start-item.submitted)*1000
        self.host_worker_ms[rank] += (host_end-host_start)*1000
        self.local_KV_staging_bytes[rank] += (2*(stop-first)*2*16*(item.position+1)*128*2
                                               if part.graph_mode and item.step>0 else 0)
        torch.cuda.nvtx.range_pop()
        if self.trace:
            anchor, anchor_host, uncertainty = self.anchors[rank]
            event = dict(rank=rank, physical_GPU=rank+4, id=item.rid, step=item.step, loop=loop,
                layer_start=first, layer_end=stop, q_tokens=hidden.shape[1],
                queue_ms=(host_start-item.submitted)*1000, host_start_s=host_start-self.origin,
                host_end_s=host_end-self.origin, transfer_kind=transfer_kind, source_rank=source,
                transfer_bytes=transfer_bytes, transfer_ms=a.elapsed_time(b),
                compute_ms=b.elapsed_time(c), head_ms=c.elapsed_time(d),
                GPU_start_s=anchor_host-self.origin+anchor.elapsed_time(b)/1000,
                GPU_end_s=anchor_host-self.origin+anchor.elapsed_time(c)/1000,
                GPU_stage_start_s=anchor_host-self.origin+anchor.elapsed_time(a)/1000,
                GPU_stage_end_s=anchor_host-self.origin+anchor.elapsed_time(d)/1000,
                anchor_uncertainty_ms=uncertainty,
                compute_scope='CUDA stream bracket. Includes launch gaps, not exclusive kernel service.')
            with self.lock:
                self.events.append(event)
        else:
            # Count bytes independently of detailed tracing. No host hidden-state copy.
            with self.lock:
                self.transfer_counts[rank][transfer_kind or 'local'] += 1
                self.transfer_bytes[rank][transfer_kind or 'local'] += transfer_bytes
        if terminal:
            token_id = int(token.item())  # Explicit token/control D2H, not hidden-state transport.
            if self.correctness:
                part.snapshots[item.rid] = dict(logits=logits.detach().float().cpu(),
                                               hidden=hidden[:, -1:, :].detach().float().cpu())
            self.completed.put(('token', item, token, token_id, time.perf_counter()))
        else:
            next_rank = self.path[item.visit + 1][0]
            self.ready[next_rank].put(Item(item.rid, item.step, item.position, item.visit + 1,
                                         hidden, time.perf_counter()))

    def run(self, requests, depth=4, output_len=8, trace=False, correctness=False, stop_eos=False,
            graphs=False, cache_capacity=None):
        import collections
        self.path = route(self.placement, depth)
        self.correctness, self.trace = correctness, trace
        for p in self.parts:
            p.configure(requests,cache_capacity or output_len,graphs)
            p.reset()
        if self.peer_pools is not None:
            for pool in self.peer_pools:pool.prepare(requests)
        self.events = []
        self.queue_wait_ms = [0.]*4
        self.host_worker_ms = [0.]*4
        self.local_KV_staging_bytes = [0]*4
        initial_graphs=[len(p.graphs) for p in self.parts]
        self.transfer_counts = [collections.Counter() for _ in range(4)]
        self.transfer_bytes = [collections.Counter() for _ in range(4)]
        self.anchors = []
        # Workload-boundary sync only. Calibrate each GPU clock against host bounds.
        for rank in range(4):
            with torch.cuda.device(rank):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(rank)
                ev = torch.cuda.Event(enable_timing=True)
                left = time.perf_counter(); ev.record(); ev.synchronize(); right = time.perf_counter()
                self.anchors.append((ev, (left+right)/2, (right-left)*1000))
        self.origin = time.perf_counter()
        outputs = {r['id']: dict(id=r['id'], input_ids=r['input_ids'], ids=[], token_times_s=[]) for r in requests}
        for r in requests:
            payload = torch.tensor([r['input_ids']], dtype=torch.long)
            self.ready[0].put(Item(r['id'], 0, 0, 0, payload, self.origin))
        done = 0
        while done < len(requests):
            result = self.completed.get(timeout=600)
            if result[0] == 'error':
                raise RuntimeError(result[1])
            _, item, token, token_id, finish = result
            out = outputs[item.rid]
            out['ids'].append(token_id)
            out['token_times_s'].append(finish-self.origin)
            if len(out['ids']) == output_len or (stop_eos and token_id == 0):
                out['finish_reason'] = 'eos' if stop_eos and token_id == 0 else 'length'
                done += 1
            else:
                position = len(out['input_ids']) + len(out['ids']) - 1
                self.ready[0].put(Item(item.rid, item.step+1, position, 0, token, time.perf_counter()))
        wall = time.perf_counter() - self.origin
        clock_checks=[]
        if trace:
            for rank,(anchor,anchor_host,uncertainty) in enumerate(self.anchors):
                with torch.cuda.device(rank):
                    end=torch.cuda.Event(enable_timing=True)
                    left=time.perf_counter();end.record();end.synchronize();right=time.perf_counter()
                    mapped=anchor_host+anchor.elapsed_time(end)/1000
                    clock_checks.append(dict(rank=rank,end_anchor_offset_ms=(mapped-(left+right)/2)*1000,
                        initial_bound_ms=uncertainty,end_bound_ms=(right-left)*1000))
        # All requests completed. No worker has unfinished dependency work at this point.
        memory = []
        for rank, part in enumerate(self.parts):
            caches = {rid: part.cache_summary(rid) for rid in outputs if rid in part.caches}
            memory.append(dict(rank=rank, physical_GPU=rank+4, decoder_layers=part.indices,
                weights_bytes=part.weight_bytes, KV_bytes=sum(s['bytes'] for s in caches.values()),
                allocated_bytes=torch.cuda.memory_allocated(rank), reserved_bytes=torch.cuda.memory_reserved(rank),
                peak_bytes=torch.cuda.max_memory_allocated(rank), caches=caches))
        return dict(placement=self.placement, depth=depth, C=len(requests), output_len=output_len,
            input_lengths=[len(r['input_ids']) for r in requests], outputs=list(outputs.values()),
            wall_s=wall, events=self.events, memory=memory,
            admission_host_perf_s=self.origin, clock_checks=clock_checks,
            implementation='v5 exact-length full-stage graphs; common owner-local KV staging',
            transfer_counts=[dict(v) for v in self.transfer_counts],
            transfer_bytes=[dict(v) for v in self.transfer_bytes],
            rank_queue_wait_ms=self.queue_wait_ms, rank_host_worker_ms=self.host_worker_ms,
            local_KV_staging_logical_bytes=self.local_KV_staging_bytes,
            graph_captures_during_run=[len(p.graphs)-n for p,n in zip(self.parts,initial_graphs)],
            CPU_token_control='Terminal argmax scalar D2H and host queue. Returned token tensor travels GPU-to-GPU.',
            trace=trace, correctness=correctness, request_batching=False,
            backend='Official HF SDPA, one CUDA worker thread per GPU, direct peer copy',
            preallocated_receive=self.peer_pools is not None,
            precomputed_rotary=all(p.precompute_rotary for p in self.parts),
            CUDA_graph_decode=graphs, graph_capacity=cache_capacity or output_len,
            no_prompt_chunking=True)

    def close(self):
        for q in self.ready:
            q.put(None)
        for t in self.threads:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in self.threads)
