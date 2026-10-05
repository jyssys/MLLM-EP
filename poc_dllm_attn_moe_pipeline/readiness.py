"""Whole-buffer release tracing; never infer unobservable row readiness."""
import functools
import inspect
import types
import torch


class Observer:
    def __init__(self, rank):
        self.rank = rank
        self.mode = 'clean'
        self.context = None
        self.events = []
        self.routes = []
        self.call = -1
        self.layer = None
        self.snapshots = {}

    @staticmethod
    def event():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def begin(self, mode, capture=False):
        self.mode = mode
        self.capture = capture
        self.events = []
        self.routes = []
        self.calls = []
        self.call = -1
        self.origin = self.event()

    def timed(self, layer, name, fn, wait=False):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if self.mode == 'clean':
                return fn(*args, **kwargs)
            a = self.event()
            result = fn(*args, **kwargs)
            if wait:
                result[1].current_stream_wait()  # same native dependency
            b = self.event()
            self.events.append((self.call, layer, name, a, b))
            if name == 'dispatch' and self.mode == 'trace':
                self.routes.append(dict(call_id=self.call,layer=layer,
                    ids=kwargs['topk_idx'].detach().clone(),weights=kwargs['topk_weights'].detach().clone(),
                    recv_ids=result[1].detach().clone(),recv_rows=result[0].shape[0]))
            return result
        return wrapped

    def install(self, model):
        original = model.forward
        def forward(_self, input_ids=None, *args, **kwargs):
            if self.mode == 'clean':
                return original(input_ids, *args, **kwargs)
            self.call += 1
            cache = kwargs.get('past_key_values')
            positions = kwargs['position_ids']
            self.context = dict(call_id=self.call, rows=input_ids.shape[1],
                                phase='hot' if cache is not None else 'cold',
                                positions=positions.detach().clone(), cache_present=cache is not None,
                                cache_length=0 if cache is None else cache.shape[4])
            native_iteration = getattr(self,'decoder_observer',None)
            native_iteration = None if native_iteration is None else native_iteration.iteration
            self.context['native_iteration'] = None if native_iteration is None else native_iteration.copy()
            a = self.event()
            result = original(input_ids, *args, **kwargs)
            b = self.event()
            self.calls.append((self.context.copy(), a, b))
            return result
        model.forward = types.MethodType(forward, model)
        for layer in model.model.layers:
            lid = layer.layer_id
            comm = layer.layer_communicator
            for name in ('prepare_attn', 'prepare_mlp', 'postprocess_layer'):
                setattr(comm, name, self.timed(lid, name, getattr(comm, name)))
            attn = layer.attention
            attn.query_key_value.forward = self.timed(lid, 'qkv', attn.query_key_value.forward)
            attn.forward = self.timed(lid, 'attention', attn.forward)
            block = layer.mlp
            if not hasattr(block, 'deepep_dispatcher'):
                continue
            self.install_moe(block, lid)
        # Trace only the source-row all-gather owned by the active MoE call.
        import torch.distributed as dist
        gather = dist.all_gather_into_tensor
        def all_gather(output, input_, *args, **kwargs):
            if self.mode == 'clean' or self.layer is None:
                return gather(output, input_, *args, **kwargs)
            # All shared and routed contributions were enqueued on this stream.
            # Event at gather entry is safe local publication, not latent math.
            return self.timed(self.layer, 'source_all_gather', gather)(output, input_, *args, **kwargs)
        dist.all_gather_into_tensor = all_gather

    def install_moe(self, block, lid):
        dispatcher = block.deepep_dispatcher
        dispatcher.dispatch = self.timed(lid, 'dispatch', dispatcher.dispatch)
        normal = dispatcher._normal_dispatcher
        normal._combine_core = self.timed(lid, 'combine', normal._combine_core, True)
        block.experts.quant_method.apply = self.timed(lid, 'expert', block.experts.quant_method.apply)
        block.shared_experts.forward = self.timed(lid, 'shared', block.shared_experts.forward)
        block.gate.forward = self.timed(lid, 'router', block.gate.forward)
        original = block.forward
        def moe(_self, hidden_states, *args, **kwargs):
            if self.mode == 'clean':
                return original(hidden_states, *args, **kwargs)
            self.layer = lid
            a = self.event()
            snapshot = (self.capture and self.context['phase'] == 'hot' and
                        hidden_states.shape[1] == 32 and lid in (1,9,17) and lid not in self.snapshots)
            if snapshot:
                self.snapshots[lid] = dict(moe_input=hidden_states.detach().clone(), call_id=self.call)
            result = original(hidden_states, *args, **kwargs)
            b = self.event()
            self.events.append((self.call, lid, 'moe', a, b))
            if snapshot:
                self.snapshots[lid]['moe_output'] = result.detach().clone()
            self.layer = None
            return result
        block.forward = types.MethodType(moe, block)
        # Capture actual next-layer inputs/cache for component-only measurements.
        # This runs only in trace lane on one calibration request.

    def install_consumer_capture(self, model):
        for lid in (2,10,18):
            layer = model.model.layers[lid]
            original = layer.forward
            def wrapped(_self, *args, _original=original, _lid=lid, **kwargs):
                snap = self.snapshots.get(_lid-1)
                if self.mode == 'trace' and snap is not None and 'consumer' not in snap and self.call == snap['call_id']:
                    bound = inspect.signature(_original).bind(*args, **kwargs).arguments
                    def clone(v):
                        if isinstance(v, torch.Tensor):
                            return v.detach().clone()
                        if isinstance(v, (tuple, list)):
                            return type(v)(clone(x) for x in v)
                        return v
                    snap['consumer'] = {k:clone(v) for k,v in bound.items()}
                return _original(*args, **kwargs)
            layer.forward = types.MethodType(wrapped, layer)

    def finish(self):
        torch.cuda.synchronize(self.rank)
        events = [dict(call_id=c, layer=l, stage=s, start_us=self.origin.elapsed_time(a)*1000,
                       end_us=self.origin.elapsed_time(b)*1000, duration_us=a.elapsed_time(b)*1000)
                  for c,l,s,a,b in self.events]
        calls = []
        for c,a,b in self.calls:
            value = {k:v for k,v in c.items() if k != 'positions'}
            value['positions'] = c['positions'].cpu().tolist()
            value.update(start_us=self.origin.elapsed_time(a)*1000, end_us=self.origin.elapsed_time(b)*1000)
            calls.append(value)
        routes=[]
        for r in self.routes:
            ids,weights,recv=(r[k].cpu() for k in ('ids','weights','recv_ids'))
            assert bool(((recv==-1)|((recv>=0)&(recv<64))).all())
            assert ids.shape[1]==8 and bool(((ids>=0)&(ids<256)).all())
            routes.append(dict(call_id=r['call_id'],layer=r['layer'],ids=ids.tolist(),
                               weights=weights.float().tolist(),recv_rows=r['recv_rows'],owner_local_ids_valid=True))
        return dict(events=events, calls=calls,routes=routes, clock_domain=f'cuda-rank{self.rank}',
                    cross_gpu_clock_alignment='UNRESOLVED; never compare absolute offsets between ranks',
                    latent_math_completion='UNOBSERVABLE', natural_release_scope='WHOLE_BUFFER')
