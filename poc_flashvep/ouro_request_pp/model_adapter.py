"""Use the pinned official decoder, sandwich norms, RoPE, and cache update."""
import gc
import importlib
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .contract import MODEL, UUIDS
from .graph_cache import ExactCache, exact_graph_stage

GRAPH_CAPTURE_LOCK = threading.Lock()


def load_cpu():
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert model.config.num_hidden_layers == 24
    assert model.config.hidden_size == 2048
    assert model.config.num_attention_heads == model.config.num_key_value_heads == 16
    return model


def tokenizer():
    return AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True, local_files_only=True)


def nbytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


class Part:
    def __init__(self, rank, placement):
        self.rank = rank
        self.device = torch.device('cuda', rank)
        model = load_cpu()
        self.remote = importlib.import_module(type(model).__module__)
        self.config = model.config
        self.indices = list(range(rank * 6, (rank + 1) * 6)) if placement == 'layer_pp4' else list(range(24))
        self.layers = torch.nn.ModuleDict({str(i): model.model.layers[i] for i in self.indices}).to(self.device)
        # Identical auxiliary replicas in both arms. Only rank 0 embeds tokens.
        # The rank that ends a physical loop applies its original final norm and gate.
        self.embedding = model.model.embed_tokens.to(self.device)
        self.norm = model.model.norm.to(self.device)
        self.gate = model.model.early_exit_gate.to(self.device)
        self.head = model.lm_head.to(self.device)
        self.rotary = model.model.rotary_emb.to(self.device)
        self.weight_bytes = dict(core=nbytes(self.layers), embedding=nbytes(self.embedding),
                                 norm=nbytes(self.norm), gate=nbytes(self.gate), head=nbytes(self.head))
        self.weight_bytes['total'] = sum(self.weight_bytes.values())
        self.caches = {}
        self.metadata = {}
        self.snapshots = {}
        self.graphs = {}
        self.graph_mode = False
        self.capture_enabled = True
        self.graph_version = 'exact_length'
        self.precompute_rotary = False
        self.rotary_constants = {}
        self.capacity_signature = None
        del model
        gc.collect()
        actual = str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')
        assert actual == UUIDS[rank], (rank, actual)

    def configure(self, requests, capacity, graph_mode):
        signature = (tuple((r['id'],len(r['input_ids'])) for r in requests), capacity, graph_mode)
        self.graph_mode = graph_mode
        if signature != self.capacity_signature:
            self.caches.clear()
            if not graph_mode or self.graph_version != 'exact_length':
                self.graphs.clear()
            self.capacity_signature = signature
        if graph_mode:
            for r in requests:
                if r['id'] not in self.caches:
                    size=len(r['input_ids'])+capacity-1
                    self.caches[r['id']] = ExactCache(size,self.remote) if self.graph_version=='exact_length' else StaticCache(size)

    def reset(self):
        if not self.graph_mode:
            self.caches.clear()
        else:
            for cache in self.caches.values():
                if isinstance(cache,ExactCache):
                    cache.reset(self.remote)
                    continue
                cache.length = 0
                # Future positions are masked. No need to read or zero obsolete rows.
                cache.prefill = True
        self.metadata.clear()
        self.snapshots.clear()

    def stage(self, hidden, rid, step, position, loop, start, stop):
        if self.graph_mode:
            return self.graph_stage(hidden, rid, step, position, loop, start, stop)
        cache = self.caches.setdefault(rid, self.remote.UniversalTransformerCache(96))
        key = (rid, step)
        if key not in self.metadata:
            pos = torch.arange(position, position + hidden.shape[1], device=self.device)
            self.metadata[key] = pos, pos.unsqueeze(0), self.rotary(hidden, pos.unsqueeze(0))
        pos, positions, rotary = self.metadata[key]
        for layer in range(start, stop):
            hidden = self.layers[str(layer)](hidden, attention_mask=None, position_ids=positions,
                past_key_value=cache, use_cache=True, cache_position=pos,
                position_embeddings=rotary, current_ut=loop)
        if stop == 24:
            hidden = self.norm(hidden)
            self.gate(hidden)  # Original gate work. No post-hoc output selection or actual early exit.
        return hidden

    def fixed_forward(self, hidden, cache, loop, start, stop, pos, mask):
        positions = pos.unsqueeze(0)
        rotary = self.prepared_rotary if self.precompute_rotary else self.rotary(hidden, positions)
        for i in range(start, stop):
            hidden = self.layers[str(i)](hidden, attention_mask=mask, position_ids=positions,
                past_key_value=cache, use_cache=True, cache_position=pos,
                position_embeddings=rotary, current_ut=loop)
        if stop == 24:
            hidden = self.norm(hidden)
            self.gate(hidden)
        return hidden

    def graph_stage(self, hidden, rid, step, position, loop, start, stop):
        if self.graph_version=='exact_length':
            return exact_graph_stage(self,hidden,rid,step,position,loop,start,stop,GRAPH_CAPTURE_LOCK)
        cache = self.caches[rid]
        cache.prefill = step == 0
        cache.length = position + hidden.shape[1]
        if step == 0:
            pos = torch.arange(hidden.shape[1],device=self.device)
            return self.fixed_forward(hidden,cache,loop,start,stop,pos,None)
        if not self.capture_enabled:
            pos = torch.tensor([position],dtype=torch.long,device=self.device)
            mask = torch.where(torch.arange(cache.capacity,device=self.device)<=position,0.,float('-inf')).to(torch.bfloat16).view(1,1,1,-1)
            return self.fixed_forward(hidden,cache,loop,start,stop,pos,mask)
        key = (rid,loop)
        if key not in self.graphs:
            inp = torch.empty_like(hidden)
            pos = torch.tensor([position],dtype=torch.long,device=self.device)
            full_positions = torch.arange(cache.capacity,device=self.device)
            inp.copy_(hidden)
            def call():
                mask = torch.where(full_positions <= pos[0], 0., float('-inf')).to(torch.bfloat16).view(1,1,1,-1)
                return self.fixed_forward(inp,cache,loop,start,stop,pos,mask)
            # Repeated overwrite at the same current position is exact, not an append.
            with GRAPH_CAPTURE_LOCK:
                call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=torch.cuda.current_stream(self.device),capture_error_mode='thread_local'):
                    out = call()
            self.graphs[key] = inp,pos,out,graph
        inp,pos,out,graph = self.graphs[key]
        inp.copy_(hidden)
        pos.fill_(position)
        graph.replay()
        return out

    def cache_summary(self, rid):
        cache = self.caches[rid]
        if isinstance(cache,ExactCache):
            return cache.summary()
        keys = [i for i, k in enumerate(cache.key_cache) if k is not None]
        sizes = {i: list(cache.key_cache[i].shape) for i in keys}
        if isinstance(cache,StaticCache):
            sizes = {i: [1,16,cache.length,128] for i in keys}
        return dict(slots=keys, shapes=sizes,
                    bytes=sum(cache.key_cache[i].numel() * cache.key_cache[i].element_size() * 2 for i in keys),
                    distinct_storage=len({cache.key_cache[i].untyped_storage().data_ptr() for i in keys}) == len(keys))

    def snapshot(self, rid):
        c = self.caches[rid]
        if isinstance(c,ExactCache):
            return c.snapshot()
        return {i: ((k[:,:,:c.length] if isinstance(c,StaticCache) else k).detach().cpu(),
                    (c.value_cache[i][:,:,:c.length] if isinstance(c,StaticCache) else c.value_cache[i]).detach().cpu())
                for i, k in enumerate(c.key_cache) if k is not None}


class StaticCache:
    """Exact owner-local KV storage. Only graph decode attends a masked capacity."""
    def __init__(self,capacity):
        self.capacity = capacity
        self.key_cache = [None]*96
        self.value_cache = [None]*96
        self.prefill = True
        self.length = 0

    def update(self,k,v,index,metadata):
        if self.key_cache[index] is None:
            shape = list(k.shape); shape[2] = self.capacity
            self.key_cache[index] = torch.zeros(shape,dtype=k.dtype,device=k.device)
            self.value_cache[index] = torch.zeros(shape,dtype=v.dtype,device=v.device)
        pos = metadata['cache_position']
        self.key_cache[index].index_copy_(2,pos,k)
        self.value_cache[index].index_copy_(2,pos,v)
        if self.prefill:
            return self.key_cache[index][:,:,:k.shape[2]], self.value_cache[index][:,:,:v.shape[2]]
        return self.key_cache[index], self.value_cache[index]
