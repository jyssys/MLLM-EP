"""Reuse physical CUDA attention graphs across changes to paged serving batches."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from .peer_attention import AssistedAttention, KVShard, _capture, fa3


def _context_bucket(length, page):
    return max(page, 1 << (length - 1).bit_length())


class PagedGraphSlot:
    def __init__(self, owner, batch, d_bound, helper_bound):
        self.owner = owner
        self.batch, self.d_bound, self.helper_bound = batch, d_bound, helper_bound
        self.arena, self.helper = owner.d, owner.helper
        self.d = self.arena.device
        self.dim = owner.dim
        self.q = torch.zeros((batch,owner.heads,owner.dim),device=self.d,dtype=torch.bfloat16)
        self.q_offsets = torch.arange(batch + 1,device=self.d,dtype=torch.int32)
        self.d_table = torch.zeros((batch,d_bound//owner.page),device=self.d,dtype=torch.int32)
        self.d_used = torch.zeros(batch,device=self.d,dtype=torch.int32)
        self.graphs = {}
        self.metadata_updates = 0
        self.invocations = 0
        self.creation_ms = []
        self.local_stream = torch.cuda.Stream(device=self.d)
        if helper_bound:
            self.e = self.helper.device
            self.e_table = torch.zeros((batch,helper_bound//owner.page),device=self.e,dtype=torch.int32)
            self.e_used = torch.zeros(batch,device=self.e,dtype=torch.int32)
            self.e_ready = torch.cuda.Event()
            self.kernel = AssistedAttention(tuple(self.q.shape),decode_device=self.d,helper_device=self.e,
                                            num_splits=owner.num_splits)

    @staticmethod
    def replace(table, used, new_table, new_used):
        # Unused columns can remain stale. The valid used lengths only reach
        # the prefix of the table overwritten in this update.
        visible = min(table.shape[1],new_table.shape[1])
        table[:,:visible].copy_(new_table[:,:visible],non_blocking=True)
        used.copy_(new_used,non_blocking=True)

    def update(self, prepared):
        self.replace(self.d_table,self.d_used,prepared.d_table,prepared.d_used)
        if self.helper_bound:
            stream = torch.cuda.current_stream(self.e)
            if getattr(prepared,"h_ready",None) is not None:
                stream.wait_event(prepared.h_ready)
            self.replace(self.e_table,self.e_used,prepared.h_table,prepared.h_used)
            self.e_ready.record(stream)
            torch.cuda.current_stream(self.d).wait_event(self.e_ready)
        self.metadata_updates += 1

    def _build(self, layer):
        when = time.perf_counter_ns()
        cache = self.arena.cache[layer]
        local = KVShard(cache[0],cache[1],(self.d_bound,)*self.batch,None,self.d_table,self.d_used,None)
        if self.helper_bound:
            remote = self.helper.cache[layer]
            shard = KVShard(remote[0],remote[1],(self.helper_bound,)*self.batch,None,self.e_table,self.e_used,None)
            graph = self.kernel.graph(self.q,local,shard)
            value = (graph,graph.output)
        else:
            graph, result = _capture(self.d,self.local_stream,lambda: fa3(
                self.q,local,self.q_offsets,self.dim**-.5,num_splits=self.owner.num_splits
            ))
            value = (graph,result)
        self.graphs[layer] = value
        self.creation_ms.append((time.perf_counter_ns()-when)/1e6)
        return value

    def attend(self, layer, query):
        if query.shape != self.q.shape or query.device != self.d:
            raise ValueError("The live decode query no longer matches its prepared graph")
        self.q.copy_(query,non_blocking=True)
        graph, result = self.graphs.get(layer) or self._build(layer)
        if self.helper_bound:
            graph()
        else:
            graph.replay()
        self.invocations += 1
        return result

    def close(self):
        if self.helper_bound:
            for executable,_ in self.graphs.values():
                executable.close()
        self.graphs.clear()

    def receipt(self):
        return {"batch":self.batch,"local_context_ceiling":self.d_bound,
                "helper_context_ceiling":self.helper_bound,"updates":self.metadata_updates,
                "invocations":self.invocations,"captured_layers":len(self.graphs),
                "capture_ms":self.creation_ms}


class PagedGraphPool:
    """One pool per D worker. Graphs own stable metadata and their physical arenas."""

    def __init__(self,d,helper,*,heads,dim,max_batch=32,max_bundles=20,max_context=65536,
                 num_splits=0,fixed_d_context=None,fixed_helper_context=None):
        self.d, self.helper = d, helper
        self.heads,self.dim = heads,dim
        self.page = d.cache.shape[3]
        if self.page < 1 or self.page & (self.page - 1):
            raise ValueError("Graph bucket extents require power-of-two KV pages")
        if helper is not None and self.page != helper.cache.shape[3]:
            raise ValueError("Graph D and helper KV pages must share their height")
        if min(max_batch,max_bundles,max_context) < 1:
            raise ValueError("Graph limits must be positive")
        for bound in (fixed_d_context,fixed_helper_context):
            if bound is not None and (bound < self.page or bound % self.page or bound > max_context):
                raise ValueError("Fixed graph cutoffs must be page-aligned and fit max_context")
        self.max_batch,self.max_bundles,self.max_context = max_batch,max_bundles,max_context
        self.fixed_d_context = fixed_d_context
        self.fixed_helper_context = fixed_helper_context
        self.num_splits = num_splits
        self.slots, self.fallbacks = {}, Counter()
        self.last_fallback = None
        self.prewarm_stats = None

    def prepare(self, prepared):
        d_lengths, h_lengths = prepared.d_lengths, prepared.h_lengths
        batch = len(d_lengths)
        self.last_fallback = None
        if batch != len(h_lengths) or batch < 1 or min(d_lengths) < 1:
            raise ValueError("A serving graph needs one valid local row per query")
        split = any(h_lengths)
        if batch > self.max_batch:
            self.last_fallback = "batch_limit"
        elif split and not all(h_lengths):
            self.last_fallback = "mixed_empty_helper"
        elif split and self.helper is None:
            raise ValueError("Cannot attach a live remote page table without a helper arena")
        else:
            local_ceiling = self.fixed_d_context or _context_bucket(max(d_lengths),self.page)
            helper_ceiling = (self.fixed_helper_context or _context_bucket(max(h_lengths),self.page)) if split else 0
            key = (batch,local_ceiling,helper_ceiling)
            if (max(local_ceiling,helper_ceiling) > self.max_context or max(d_lengths) > local_ceiling
                    or max(h_lengths) > helper_ceiling):
                self.last_fallback = "context_limit"
            elif key not in self.slots and len(self.slots) >= self.max_bundles:
                self.last_fallback = "bundle_limit"
            else:
                if key not in self.slots:
                    self.slots[key] = PagedGraphSlot(self,*key)
                slot = self.slots[key]
                slot.update(prepared)
                return slot
        self.fallbacks[self.last_fallback] += 1
        return None

    @torch.inference_mode()
    def prewarm(self,batches,*,include_local=True,include_split=False):
        """Capture fixed-ceiling plans using one dummy page before admitting live KV."""
        batches = tuple(dict.fromkeys(batches))
        if not batches or min(batches) < 1 or max(batches) > self.max_batch:
            raise ValueError("Warmup batches must fit the graph batch limit")
        if not (include_local or include_split) or self.fixed_d_context is None:
            raise ValueError("Graph warmup requires a fixed D ceiling and a route")
        if include_split and (self.helper is None or self.fixed_helper_context is None):
            raise ValueError("Split graph warmup requires fixed helper ceiling and physical helper")
        if self.slots or self.prewarm_stats is not None:
            raise ValueError("Prewarm each graph pool once before starting its admission queue")
        if len(batches)*(int(include_local)+int(include_split)) > self.max_bundles:
            raise ValueError("Warmup would exceed the graph bundle limit")
        arenas = (self.d,self.helper) if include_split else (self.d,)
        if any(getattr(arena,"owners",None) for arena in arenas):
            raise ValueError("Do not zero a dummy page after KV admission has started")
        started = time.perf_counter_ns()
        for arena in arenas:
            arena.cache[:,:,0].zero_()
        d = self.d.device
        e = self.helper.device if include_split else None
        with torch.cuda.device(d):
            for batch in batches:
                d_table = torch.zeros((batch,1),device=d,dtype=torch.int32)
                d_used = torch.ones(batch,device=d,dtype=torch.int32)
                query = torch.zeros((batch,self.heads,self.dim),device=d,dtype=torch.bfloat16)
                prepared = SimpleNamespace(d_lengths=(1,)*batch,h_lengths=(0,)*batch,
                                           d_table=d_table,d_used=d_used)
                for split in (False,True):
                    if (split and not include_split) or (not split and not include_local):
                        continue
                    if split:
                        prepared.h_lengths = (1,)*batch
                        prepared.h_table = torch.zeros((batch,1),device=e,dtype=torch.int32)
                        prepared.h_used = torch.ones(batch,device=e,dtype=torch.int32)
                    slot = self.prepare(prepared)
                    if slot is None:
                        raise RuntimeError(f"Dummy prepared graph unexpectedly fell back: {self.last_fallback}")
                    for layer in range(self.d.cache.shape[0]):
                        slot.attend(layer,query)
                    # Exclude dummy metadata and invocations from live batch counts.
                    slot.metadata_updates = slot.invocations = 0
            torch.cuda.synchronize(d)
        self.prewarm_stats = {
            "batches":batches,"include_local":include_local,"include_split":include_split,
            "physical_page":0,"wall_ms":(time.perf_counter_ns()-started)/1e6,
            "bundles":len(self.slots),"layers_each":self.d.cache.shape[0],
        }

    def receipt(self):
        return {"max_batch":self.max_batch,"max_bundles":self.max_bundles,
                "max_context":self.max_context,"num_splits":self.num_splits,
                "fixed_d_context":self.fixed_d_context,"fixed_helper_context":self.fixed_helper_context,
                "prewarm":self.prewarm_stats,
                "slots":[slot.receipt() for slot in self.slots.values()],"fallbacks":dict(self.fallbacks)}

    def close(self):
        torch.cuda.synchronize(self.d.device)
        for slot in self.slots.values():
            slot.close()


@torch.inference_mode()
def smoke(out,decode,helper):
    d, e = torch.device("cuda",decode),torch.device("cuda",helper)
    batch, heads, kv_heads, dimension, layers, blocks, page = 4,28,4,128,2,24,128
    d_state,h_state = (
        SimpleNamespace(device=device,cache=torch.randn((layers,2,blocks,page,kv_heads,dimension),
                                                        dtype=torch.bfloat16,device=device))
        for device in (d,e)
    )
    query = torch.randn((batch,heads,dimension),dtype=torch.bfloat16,device=d)
    family = PagedGraphPool(d_state,h_state,heads=heads,dim=dimension,max_context=512,
                            fixed_d_context=256,fixed_helper_context=256)
    family.prewarm((batch,),include_local=True,include_split=True)
    eager = AssistedAttention(tuple(query.shape),decode_device=d,helper_device=e)
    results = []
    layouts = (
        ([170,160,180,145],[168,163,150,140],0),
        ([144,176,168,152],[142,189,164,160],8),
        ([180,180,180,180],[0,0,0,0],0),
        ([144,160,152,160],[0,0,0,0],8),
    )
    for d_lengths, e_lengths, start in layouts:
        d_table = torch.arange(start,start+2*batch,device=d,dtype=torch.int32).view(batch,2)
        e_table = torch.arange(start,start+2*batch,device=e,dtype=torch.int32).view(batch,2)
        dd = torch.tensor(d_lengths,dtype=torch.int32,device=d)
        ee = torch.tensor(e_lengths,dtype=torch.int32,device=e)
        page_view = SimpleNamespace(d_lengths=d_lengths,h_lengths=e_lengths,
                                    d_table=d_table,d_used=dd,h_table=e_table,h_used=ee)
        graph = family.prepare(page_view)
        if graph is None:
            raise RuntimeError("Expected constructed graph for the prepared nonempty or local-only case")
        for layer in range(layers):
            query.add_(.0625)
            created = graph.attend(layer,query).clone()
            local = KVShard.paged(d_state.cache[layer,0],d_state.cache[layer,1],d_lengths,d_table,dd)
            remote = KVShard.paged(h_state.cache[layer,0],h_state.cache[layer,1],e_lengths,e_table,ee)
            reference = eager.split(query,local,remote)
            difference = (created.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-20)
            relative_l2 = float(difference)
            if relative_l2 > .006 or not bool(torch.isfinite(created).all()):
                raise RuntimeError(f"Adaptive block table update differs from independently issued FA3: {relative_l2}")
            results.append({"d_lengths":d_lengths,"helper_lengths":e_lengths,"page_table_offset":start,
                            "layer":layer,"relative_l2":relative_l2,"bitwise":bool(torch.equal(created,reference))})
    # Mixed route must be handled by the empty-safe eager fallback.
    page_view.h_lengths = [128,0,128,132]
    expected = family.prepare(page_view)
    if expected is not None or family.last_fallback != "mixed_empty_helper":
        raise RuntimeError("Expected mixed-helper batch fallback")
    status = family.receipt()
    family.close()
    with out.open("x") as stream:
        json.dump({"status":"PASS","layers":layers,"dynamic_placements":results,"graph_stats":status},stream,indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-output",type=Path,required=True)
    parser.add_argument("--decode-device",type=int,default=0)
    parser.add_argument("--helper-device",type=int,default=1)
    args = parser.parse_args()
    with torch.cuda.device(args.decode_device):
        smoke(args.smoke_output,args.decode_device,args.helper_device)


if __name__ == "__main__":
    main()
