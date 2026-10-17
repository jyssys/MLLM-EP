"""Serial numerical emulation of the causal, frozen-input concurrent-round DAG.

The official forward and attention/cache implementation remain unchanged.
Hooks replace stage inputs, not weights, routes, cache slots, or attention masks.
This module provides no physical concurrency or latency speedup evidence.
"""
import time
import torch


class StageEmulator:
    def __init__(self, model, arm='exact', coefficients=None, stages=(1,2,3), rounds=(1,2,3), capture=False):
        self.model = model
        self.arm = arm
        self.coefficients = coefficients
        self.stages = set(stages)
        self.rounds = set(rounds) | ({0} if arm == 'w0' else set())
        self.capture = capture
        self.events = []
        self.captures = []
        self.gpu_events = []
        self.anchor = None
        self.enabled = True
        self.n = model.config.num_hidden_layers
        assert self.n == 24 and model.config.total_ut_steps == 4
        self.handles = [model.model.layers[i].register_forward_pre_hook(self.hook(j), with_kwargs=True)
                        for j,i in enumerate(range(0,self.n,6))]
        self.handles += [model.model.layers[i].register_forward_hook(self.output_hook(j))
                         for j,i in enumerate(range(5,self.n,6))]

    def close(self):
        for handle in self.handles:
            handle.remove()

    def gpu_event(self, position, r, stage, kind, value):
        if not self.capture or not value.is_cuda or len(self.gpu_events)>=512:
            return
        if self.anchor is None:
            self.anchor=torch.cuda.Event(enable_timing=True)
            self.anchor.record()
        event=torch.cuda.Event(enable_timing=True)
        event.record()
        self.gpu_events.append((event,dict(token_position=position,round=r,stage=stage,kind=kind)))

    def resolved_gpu_trace(self):
        if not self.gpu_events:
            return []
        torch.cuda.synchronize()
        return [metadata|dict(GPU_ms_from_local_anchor=self.anchor.elapsed_time(event),
                              clock='Single GPU CUDA event. Not a cross-GPU clock.')
                for event,metadata in self.gpu_events]

    def output_hook(self, stage):
        def completed(module,args,value):
            # x[r,j] means COMPUTED output of F_j, not the prediction fed to F_(j+1).
            # Reusing the latter would freeze stale states and discard completed work.
            self.outputs[stage+1]=value.detach().clone()
            self.gpu_event(self.position,self.recurrence,stage,'stage_output_ready',self.outputs[stage+1])
        return completed

    def hook(self, stage):
        def replace(module, args, kwargs):
            value = args[0]
            r = kwargs['current_ut']
            position = int(kwargs['cache_position'][0])
            if stage == 0:
                self.gpu_event(position,r,0,'round_input_ready',value)
                self.position,self.recurrence=position,r
                if r == 0:
                    self.current = {}
                    self.previous = None
                    self.outputs = {}
                    self.round_inputs = []
                    self.frozen = {}
                else:
                    self.previous = self.outputs
                    self.outputs = {}
                    self.current = {}
                self.round_inputs.append(value.detach().clone())
                active = (self.enabled and value.shape[1] == 1 and self.arm != 'exact'
                          and r in self.rounds and (r > 0 or self.arm == 'w0'))
                # All relaxed inputs are computed before any stage in this round.
                self.frozen = {}
                if active:
                    for j in sorted(self.stages):
                        if r == 0:
                            beta = self.coefficients['w0_beta'][j-1]
                            bias = torch.tensor(self.coefficients['w0_bias'][j-1], device=value.device, dtype=torch.float32)
                            predicted = (value.float()*beta + bias).to(value.dtype)
                            sources = [(position,0,0,'embedding')]
                        else:
                            previous = self.previous[j]
                            if self.arm == 'stale':
                                predicted = previous.clone()
                            else:
                                alpha = self.coefficients['alpha'][r-1][j-1]
                                predicted = (previous.float() + alpha*(value.float()-self.round_inputs[r-1].float())).to(value.dtype)
                            sources = [(position,r-1,j-1,'previous_preceding_stage_output'),
                                       (position,r,0,'current_round_input'), (position,r-1,0,'previous_round_input')]
                        self.frozen[j] = predicted
                        self.gpu_event(position,r,j,'prediction_ready',predicted)
                        if len(self.events) < 512:
                            self.events.append(dict(token_position=position,round=r,stage=j,
                                source_tensors=sources,freeze_sequence=len(self.events),
                                host_enqueue_ns=time.monotonic_ns(),
                                availability='Same-stream program order. No GPU completion timestamp claim.',
                                prediction_ready_before_stage1=True))
            if stage in self.frozen:
                value = self.frozen[stage]
                args = (value,)+args[1:]
            self.current[stage] = value.detach().clone()
            self.gpu_event(position,r,stage,'stage_input_copy_ready',self.current[stage])
            self.gpu_event(position,r,stage,'consumer_launch_boundary',value)
            if self.capture:
                # Bounded last-position summaries are diagnostic copies only.
                self.captures.append(dict(round=r,stage=stage,position=position,
                    value=value[:,-1,:].detach().float().cpu()))
            return args, kwargs
        return replace


def cache_signature(cache):
    assert len(cache.key_cache) == 96 and len(cache.value_cache) == 96
    lengths = [cache.get_seq_length(i) for i in range(96)]
    assert len(set(lengths)) == 1
    assert len({key.untyped_storage().data_ptr() for key in cache.key_cache}) == 96
    return dict(slots=96, lengths=lengths, depth_index='24*recurrent_depth+physical_layer', distinct_key_storage=True)


@torch.inference_mode()
def generate(model, tokenizer, prompt_ids, cap, emulator=None):
    """Exact full-depth prefill, followed by free-running candidate decode."""
    prompt = torch.tensor([prompt_ids], device='cuda')
    past = None
    ids = []
    start = time.perf_counter()
    if emulator is not None:
        emulator.enabled = False
    result = model(prompt, use_cache=True, exit_at_step=3, logits_to_keep=1)
    if emulator is not None:
        emulator.enabled = True
    while len(ids) < cap:
        token = int(result.logits[0,-1].argmax())
        ids.append(token)
        if token == tokenizer.eos_token_id or len(ids) == cap:
            break
        result = model(torch.tensor([[token]], device='cuda'), past_key_values=result.past_key_values,
                       use_cache=True, exit_at_step=3, logits_to_keep=1)
    torch.cuda.synchronize()
    return dict(ids=ids,text=tokenizer.decode(ids,skip_special_tokens=False),
        generated_tokens=len(ids), finish_reason='eos' if ids[-1]==tokenizer.eos_token_id else 'length',
        wall_ms=(time.perf_counter()-start)*1000,
        cache=cache_signature(result.past_key_values),
        exact_prefill=True, approximate_KV_persisted=emulator is not None and emulator.arm!='exact',
        scope='Single-GPU functional emulation. Wall time is not optimized parallel serving performance.')


@torch.inference_mode()
def legacy_compact_batch(model, tokenizer, prompts, cap, emulator=None, on_complete=None):
    """Left-padded exact prefill, fresh candidate KV, and EOS row compaction.

    A completed row leaves the cache before the next model invocation.
    Every arm uses the same frozen prompt groups. No reference KV is injected.
    """
    width=max(map(len,prompts))
    inp=torch.tensor([[tokenizer.eos_token_id]*(width-len(ids))+ids for ids in prompts],device='cuda')
    mask=torch.tensor([[0]*(width-len(ids))+[1]*len(ids) for ids in prompts],device='cuda')
    positions=mask.cumsum(-1)-1
    positions.masked_fill_(mask==0,1)
    if emulator is not None:
        emulator.enabled=False
    start=time.perf_counter()
    result=model(inp,attention_mask=mask,position_ids=positions,
                 use_cache=True,exit_at_step=3,logits_to_keep=1)
    if emulator is not None:
        emulator.enabled=True
    live=list(range(len(prompts)))
    sequences=[[] for _ in prompts]
    completed={}
    while live:
        tokens=result.logits[:,-1].argmax(-1).tolist()
        keep=[]
        for i,(original,token) in enumerate(zip(live,tokens)):
            sequences[original].append(token)
            if token==tokenizer.eos_token_id or len(sequences[original])==cap:
                ids=sequences[original]
                completed[original]=dict(ids=ids,text=tokenizer.decode(ids,skip_special_tokens=False),
                    generated_tokens=len(ids),finish_reason='eos' if token==tokenizer.eos_token_id else 'length',
                    wall_ms=(time.perf_counter()-start)*1000,exact_prefill=True,
                    approximate_KV_persisted=emulator is not None and emulator.arm!='exact',
                    cache_slots=96,KV_writes_after_EOS=0,baseline_batch_size=len(prompts),
                    scope='Batched single-GPU numerical emulation. Not physical pipeline timing.')
                if on_complete is not None:
                    on_complete(original,completed[original])
            else:
                keep.append(i)
        if not keep:
            break
        if len(sequences[live[keep[0]]])%512==0:
            print('decode progress',len(sequences[live[keep[0]]]),'of cap',cap,'active',len(keep),flush=True)
        past=result.past_key_values
        if len(keep)!=len(live):
            selection=torch.tensor(keep,device='cuda')
            past.reorder_cache(selection)
            mask=mask.index_select(0,selection)
        live=[live[i] for i in keep]
        inp=torch.tensor([[tokens[i]] for i in keep],device='cuda')
        mask=torch.cat([mask,torch.ones((len(live),1),device='cuda',dtype=mask.dtype)],dim=1)
        positions=mask.sum(-1,keepdim=True)-1
        result=model(inp,attention_mask=mask,position_ids=positions,past_key_values=past,
                     use_cache=True,exit_at_step=3,logits_to_keep=1)
    return [completed[i] for i in range(len(prompts))]


@torch.inference_mode()
def generate_batch(model, tokenizer, prompts, cap, emulator=None, on_complete=None):
    """Use the official generation driver for exact and predicted-input arms.

    The batch remains fixed, as in unmodified GenerationMixin. Native padding
    after EOS is not a new logical request token. Those padding KV entries are
    unused and never transferred to another request. There is no row compaction.
    """
    from transformers import StoppingCriteria, StoppingCriteriaList
    width=max(map(len,prompts))
    batch=len(prompts)
    inputs=torch.tensor([[tokenizer.eos_token_id]*(width-len(row))+row for row in prompts],device='cuda')
    mask=torch.tensor([[0]*(width-len(row))+[1]*len(row) for row in prompts],device='cuda')
    records={}
    start=time.perf_counter()
    class Observe(StoppingCriteria):
        def __call__(self,input_ids,scores,**kwargs):
            count=input_ids.shape[1]-width
            tokens=input_ids[:,-1].tolist()
            for i,token in enumerate(tokens):
                if i in records:
                    continue
                if token==tokenizer.eos_token_id or count>=cap:
                    ids=input_ids[i,width:].tolist()
                    record=dict(ids=ids,text=tokenizer.decode(ids,skip_special_tokens=False),
                        generated_tokens=len(ids),finish_reason='eos' if token==tokenizer.eos_token_id else 'length',
                        wall_ms=(time.perf_counter()-start)*1000,exact_prefill=True,
                        approximate_KV_persisted=emulator is not None and emulator.arm!='exact',
                        cache_slots=96,logical_token_writes_after_EOS=0,
                        native_physical_inactive_padding_KV=True,native_fixed_batch=True,
                        baseline_batch_size=batch,
                        scope='Official HF generation with boundary hooks. Single-GPU functional emulation, not parallel serving timing.')
                    records[i]=record
                    if on_complete is not None:
                        on_complete(i,record)
            if count%512==0:
                print('native decode progress',count,'of cap',cap,'unfinished',batch-len(records),flush=True)
            # This observer must never change the official stopping decision.
            return torch.zeros(batch,dtype=torch.bool,device=input_ids.device)
    handle=None
    if emulator is not None:
        emulator.enabled=False
        def enable_after_prefill(module,args,value):
            emulator.enabled=True
        handle=model.register_forward_hook(enable_after_prefill)
    try:
        result=model.generate(inputs,attention_mask=mask,max_new_tokens=cap,do_sample=False,
            eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id,
            use_cache=True,exit_at_step=3,return_dict_in_generate=True,
            stopping_criteria=StoppingCriteriaList([Observe()]))
    finally:
        if handle is not None:
            handle.remove()
    signature=cache_signature(result.past_key_values)
    assert len(records)==batch
    for i in range(batch):
        native_ids=result.sequences[i,width:].tolist()
        if tokenizer.eos_token_id in native_ids:
            native_ids=native_ids[:native_ids.index(tokenizer.eos_token_id)+1]
        assert native_ids==records[i]['ids']
    return [records[i] for i in range(batch)]
