"""Official MultiBD engine with a bounded physical-EP bridge.

One R per process session. The pinned scheduler/sampler/cache are unchanged.
Bring-up failures are retained; this is not a new serving implementation.
"""
import argparse
import gzip
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
from poc_multibd_moe_ep.contract import *


def instrument_model(model, runner):
    from diffulex.moe.layer.ep_impl import EPFusedMoE
    from diffulex.moe.layer.tp_impl import TPFusedMoE
    rank=runner.rank
    mode=os.environ['POC_MULTIBD_MODE']
    dest=Path(os.environ['POC_MULTIBD_OUTPUT'])
    ownership=[]
    tp_ownership=[]
    records=[]
    forward_inputs=[]
    hooks=[]
    runner._poc_mode=mode
    for name, module in model.named_modules():
        if isinstance(module,TPFusedMoE):
            assert module.tp_size==4 and module.num_local_experts==64
            assert module.w13.dtype==torch.bfloat16 and module.w2.dtype==torch.bfloat16
            entry=dict(name=name,owner_ids=list(range(module.local_expert_start,module.local_expert_end)),
                w13=list(module.w13.shape),w2=list(module.w2.shape),
                expert_dtype=str(module.w13.dtype),
                tensor_bytes=module.w13.numel()*module.w13.element_size()+
                             module.w2.numel()*module.w2.element_size(),
                backend='TPFusedMoE full source batch + partial-output all-reduce; no A2A dispatcher')
            if name in ['model.layers.1.mlp','model.layers.19.mlp']:
                entry['weight_probe']={
                    'gate_proj':module.w13[0,0,:8].float().cpu().tolist(),
                    'up_proj':module.w13[0,module.intermediate_size,:8].float().cpu().tolist(),
                    'down_proj':module.w2[0,0,:8].float().cpu().tolist()}
            tp_ownership.append(entry)
        if not isinstance(module,EPFusedMoE):
            continue
        assert module.ep_size==4 and module.num_local_experts==64
        assert module.w13.dtype==torch.bfloat16 and module.w2.dtype==torch.bfloat16
        assert module.active_expert_ids==list(range(rank*64,(rank+1)*64))
        ownership.append(dict(name=name, owner_ids=module.active_expert_ids,
            w13=list(module.w13.shape),w2=list(module.w2.shape),
            expert_dtype=str(module.w13.dtype),
            tensor_bytes=module.w13.numel()*module.w13.element_size()+
                         module.w2.numel()*module.w2.element_size(),
            backend=type(module.dispatcher).__name__))
        if name in ['model.layers.1.mlp','model.layers.19.mlp']:
            # Outside inference timing: exact BF16 samples from actually loaded
            # owner tensors, checked later against the pinned safetensors shard.
            ownership[-1]['weight_probe']={
                'gate_proj':module.w13[0,0,:8].float().cpu().tolist(),
                'up_proj':module.w13[0,module.intermediate_size,:8].float().cpu().tolist(),
                'down_proj':module.w2[0,0,:8].float().cpu().tolist()}
        if mode=='clean':
            continue
        dispatcher=module.dispatcher
        dispatch=dispatcher.dispatch
        expert=module._run_dispatched_experts
        combine=dispatcher.combine
        forward=module.forward
        state={}
        def wrap_dispatch(hidden,ids,weights,*,_fn=dispatch,_state=state,_name=name):
            entry=dict(layer=_name, source_rank=rank, rows=hidden.shape[0])
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();output=_fn(hidden,ids,weights);b.record()
            entry['dispatch_events']=(a,b)
            entry['recv_rows']=output.metadata.native_recv_num_tokens
            entry['recv_paths']=output.metadata.total_recv_slots
            if runner._poc_mode=='trace':
                entry['ids']=ids.detach().clone()
                entry['weights']=weights.detach().clone()
            records.append(entry);_state['entry']=entry
            entry['forward_index']=len(forward_inputs)-1
            return output
        def wrap_expert(dispatched,weight_dtype,*,_fn=expert,_state=state):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();output=_fn(dispatched,weight_dtype);b.record()
            _state['entry']['expert_events']=(a,b)
            return output
        def wrap_combine(outputs,metadata,*,_fn=combine,_state=state):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();result=_fn(outputs,metadata);b.record()
            _state['entry']['combine_events']=(a,b)
            return result
        def wrap_forward(hidden,*,_fn=forward,_state=state):
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();result=_fn(hidden);b.record()
            _state['entry']['moe_total_events']=(a,b)
            return result
        hooks.extend([(dispatcher,'dispatch',dispatch,wrap_dispatch),
                      (module,'_run_dispatched_experts',expert,wrap_expert),
                      (dispatcher,'combine',combine,wrap_combine),
                      (module,'forward',forward,wrap_forward)])
        dispatcher.dispatch=wrap_dispatch
        module._run_dispatched_experts=wrap_expert
        dispatcher.combine=wrap_combine
        module.forward=wrap_forward
    if os.environ['POC_MULTIBD_BACKEND']=='ep4':
        assert len(ownership)==19,'All sparse layers must be genuinely EP-owned'
    else:
        assert len(tp_ownership)==19 and not ownership,'TP control must have no DeepEP sparse layer'
    write(dest/f'ownership.rank{rank}.json',dict(rank=rank,physical_gpu=GPUS[rank],
        actual_cuda_uuid=str(torch.cuda.get_device_properties(rank).uuid),
        sparse_layers=ownership,tp_control_sparse_layers=tp_ownership,full_model_loaded=True,
        checkpoint=MODEL_ID,revision=MODEL_REVISION))
    original_exit=runner.exit
    def flush_records(tag='final'):
        torch.cuda.synchronize()
        finalized=[]
        for i, entry in enumerate(records):
            row={k:v for k,v in entry.items() if not k.endswith('_events') and
                 k not in ['ids','weights']}
            row['call_layer_index']=i
            for component in ['dispatch','expert','combine','moe_total']:
                events=entry[component+'_events']
                row[component+'_ms']=events[0].elapsed_time(events[1])
            if 'ids' in entry:
                row['ids']=entry['ids'].cpu().tolist()
                row['weights']=entry['weights'].cpu().tolist()
            finalized.append(row)
        inputs=[]
        for item in forward_inputs:
            inputs.append({k:v.cpu().tolist() if isinstance(v,torch.Tensor) else v
                           for k,v in item.items()})
        value=dict(records=finalized,forward_inputs=inputs,
            timing_rule='rank-local CUDA events; final analysis chooses one cumulative-critical rank',
            full_hidden_router_all_reduce_and_shared_experts='outside these D/E/C intervals')
        target=dest/tag;target.mkdir(parents=True,exist_ok=True)
        with gzip.open(target/f'{runner._poc_mode}.events.rank{rank}.json.gz','wt') as out:
            json.dump(value,out)
    def flush_exit():
        flush_records();original_exit()
    runner.exit=flush_exit
    runner._poc_records=records
    runner._poc_forward_inputs=forward_inputs
    runner._poc_flush=flush_records
    runner._poc_hooks=hooks


def begin_probe(runner,mode=None,deterministic=None):
    # The upstream control RPC returns after rank-0 handling, not after every
    # worker has finished flushing. Probe barriers are OUTSIDE measured inference.
    torch.cuda.synchronize()
    if deterministic is not None:
        # DeepEP allocates scratch on one stream and writes it on its comm
        # stream. PyTorch's automatic poison-fill races with those writes.
        # Keep the native allocation contract; select deterministic operators
        # without introducing asynchronous integer-max/NaN scratch fills.
        torch.utils.deterministic.fill_uninitialized_memory=False
        torch.use_deterministic_algorithms(deterministic)
    if mode is not None:
        assert mode in ['clean','timing','trace']
        runner._poc_mode=mode
        for obj,name,original,wrapped in runner._poc_hooks:
            setattr(obj,name,original if mode=='clean' else wrapped)
    runner._poc_records.clear();runner._poc_forward_inputs.clear()
    torch.distributed.barrier(device_ids=[runner.rank])


def end_probe(runner,tag):
    runner._poc_flush(tag)
    torch.distributed.barrier(device_ids=[runner.rank])


def install_worker_hooks():
    if os.environ.get('POC_MULTIBD_DETERMINISTIC')=='1':
        torch.utils.deterministic.fill_uninitialized_memory=False
        torch.use_deterministic_algorithms(True)
    from poc_multibd_moe_ep.bounded_bridge import repair_normal_owner_local_ids
    repair_normal_owner_local_ids()
    from diffulex.engine.model_runner import ModelRunnerBase
    original=ModelRunnerBase.load_model
    def load(self,config):
        model=original(self,config)
        instrument_model(model,self)
        return model
    ModelRunnerBase.load_model=load
    ModelRunnerBase.begin_probe=begin_probe
    ModelRunnerBase.end_probe=end_probe
    from diffulex.strategy_template.multi_block.engine.model_runner import MultiBlockModelRunnerTemplate
    forward=MultiBlockModelRunnerTemplate.run_model_multi_block
    prepare=MultiBlockModelRunnerTemplate.prepare_chunked_prefill_multi_block
    def wrapped_prepare(self,reqs):
        self._poc_before_state=[]
        if self._poc_mode=='trace':
            for req in reqs:
                buffer=req.dllm_block_buffer
                self._poc_before_state.append(dict(active_blocks=len(buffer.active_blocks),
                    valid_blocks=len(buffer.valid_blocks),live_mask_tokens=sum(b.num_mask_tokens for b in buffer.valid_blocks),
                    slots=[dict(block_id=b.block_id,status=str(b.status),
                        is_dummy=b.is_dummy,is_active=b.is_active,live_masks=b.num_mask_tokens)
                        for b in buffer.dllm_blocks]))
        return prepare(self,reqs)
    MultiBlockModelRunnerTemplate.prepare_chunked_prefill_multi_block=wrapped_prepare
    def wrapped_forward(self,input_ids,positions):
        values=dict(num_tokens=input_ids.numel())
        if self._poc_mode=='trace':
            values.update(input_ids=input_ids.detach().clone(),positions=positions.detach().clone())
            values['before_forward_state']=self._poc_before_state
        self._poc_forward_inputs.append(values)
        return forward(self,input_ids,positions)
    MultiBlockModelRunnerTemplate.run_model_multi_block=wrapped_forward


def patched_worker(config,rank,event):
    from diffulex.engine.engine import _set_parent_death_signal
    _set_parent_death_signal()
    install_worker_hooks()
    from diffulex.engine.model_runner import AutoModelRunner
    AutoModelRunner.from_config(config,rank,event)


def main():
    p=argparse.ArgumentParser();p.add_argument('--R',type=int,choices=[1,2,3,4],required=True)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=['clean','timing','trace','paired'],default='clean')
    p.add_argument('--backend',choices=['ep4','tp4'],default='ep4')
    p.add_argument('--limit',type=int,default=1);p.add_argument('--max-tokens',type=int,default=64)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--determinism-probe',action='store_true')
    p.add_argument('--split',choices=['calibration','held_out'],default='calibration')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==MASK
    for rank in range(4):
        assert str(torch.cuda.get_device_properties(rank).uuid).removeprefix('GPU-')==UUIDS[rank]
    dest=Path(a.output);dest.mkdir(parents=True,exist_ok=True)
    os.environ.update(POC_MULTIBD_MODE=a.mode,POC_MULTIBD_OUTPUT=str(dest),POC_MULTIBD_BACKEND=a.backend)
    sys.path.insert(0,str(UPSTREAM))
    from poc_multibd_moe_ep.bounded_bridge import enable_config_ep
    enable_config_ep();install_worker_hooks()
    import diffulex.engine.engine as engine
    engine._run_model_runner_worker=patched_worker
    from diffulex import Diffulex,SamplingParams
    pins=json.loads((ROOT/'artifacts/pins.json').read_text())
    for file in pins['model_files']:
        if file['name'].endswith('.safetensors'):
            path=Path(pins['model_path'])/file['name']
            assert path.is_file() and path.stat().st_size==file['bytes'], 'Incomplete checkpoint shard: '+file['name']
    with socket.socket() as s:
        s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    llm=None
    try:
        llm=Diffulex(pins['model_path'],model_name='llada2_mini',decoding_strategy='multi_bd',
            block_size=32,page_size=32,buffer_size=a.R,sampling_mode='naive',
            mask_token_id=156895, tensor_parallel_size=4,expert_parallel_size=4 if a.backend=='ep4' else 1,
            data_parallel_size=1,device_ids=[0,1,2,3],master_port=port,
            shm_name=f'multibd_poc_{os.getpid()}',distributed_timeout_seconds=120,
            moe_dispatcher_backend='deepep' if a.backend=='ep4' else 'standard',
            deepep_mode='normal',moe_gemm_impl='triton',
            enforce_eager=True,enable_torch_compile=False,enable_full_static_runner=False,
            enable_prefix_caching=True,gpu_memory_utilization=.4,
            max_num_reqs=1,max_model_len=4096,max_num_batched_tokens=4096,
            add_block_threshold=.9,semi_complete_threshold=.9,accept_threshold=.95)
        rows=[x for x in json.loads((ROOT/'artifacts/frozen_manifest.json').read_text())['rows']
              if x['split']==a.split][a.offset:a.offset+a.limit]
        if a.smoke:
            rows=[dict(domain='diagnostic',split='smoke',selected_index=0,
                data=dict(question='Write a Python function that adds two numbers.'))]
        for index,row in enumerate(rows,start=a.offset):
            torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
            prompt=llm.tokenizer.apply_chat_template([
                dict(role='system',content='You are a helpful assistant.'),
                dict(role='user',content=row['data']['question'])],
                tokenize=True,add_generation_prompt=True)
            if not a.smoke:
                prompt+=llm.tokenizer.encode(row['data']['prefix'],add_special_tokens=False)
            # Prime this exact request before measured repeats. Warmups are
            # retained, not silently discarded or treated as independent tasks.
            if a.determinism_probe:
                assert a.backend=='ep4' and a.limit==1,'Bounded one-request diagnostic only'
                probes=[(i,'clean',True,False) for i in range(a.warmup)]
                for i in range(a.repeats):
                    settings=[False,True] if i%2==0 else [True,False]
                    probes.extend((i,'clean',False,setting) for setting in settings)
                probes.extend([(0,'trace',False,False),(0,'trace',False,True)])
            elif a.mode=='paired':
                probes=[(i,'clean',True) for i in range(a.warmup)]
                for i in range(a.repeats):
                    # Counterbalance observer ON/OFF within one loaded model.
                    lanes=['clean','timing'] if i%2==0 else ['timing','clean']
                    probes.extend((i,lane,False) for lane in lanes)
                probes.append((0,'trace',False))
            else:
                probes=[(i,a.mode,i<a.warmup) for i in range(a.warmup+a.repeats)]
            signatures=[]
            for probe in probes:
                repeat,lane,is_warmup=probe[:3]
                deterministic=probe[3] if a.determinism_probe else None
                llm.model_runner.call('begin_probe',lane if a.mode=='paired' or a.determinism_probe else None,deterministic)
                torch.cuda.synchronize();start=time.perf_counter()
                output=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=a.max_tokens,max_nfe=1024),use_tqdm=False)
                torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
                trajectories=[x.to_dict() for x in output.trajectories]
                tag=f'request{index}_'+(f'{lane}_' if a.mode=='paired' else '')+f'repeat{repeat}'
                if a.determinism_probe:tag+=f'_det{int(deterministic)}'
                if is_warmup and a.mode=='paired':tag+='_warmup'
                elif is_warmup and a.determinism_probe:tag+='_warmup'
                write(dest/f'{tag}.json',dict(R=a.R,mode=lane,backend=a.backend,
                    gpu_mask=MASK,measurement_session=os.environ.get('POC_MULTIBD_SESSION','historical_gpu2367'),
                    deterministic_ops=torch.are_deterministic_algorithms_enabled(),
                    determinism_diagnostic=a.determinism_probe,
                    request=row,repeat=repeat,is_warmup=is_warmup,
                    wall_ms=wall,prompt_ids=prompt,trajectories=trajectories,
                    bounded_smoke=a.smoke,quality_score=None))
                if not is_warmup:
                    llm.model_runner.call('end_probe',tag)
                    if not a.determinism_probe:
                        from poc_multibd_moe_ep.scripts.analyze_conversion import acceptance_signature
                        signatures.append(acceptance_signature(dict(trajectories=trajectories)))
                print('REQUEST_COMPLETE',a.R,index,repeat,wall,flush=True)
            if signatures:
                parity=all(s==signatures[0] for s in signatures)
                write(dest/f'parity_request{index}.json',dict(pass_all=parity,probes=len(signatures),
                    final_tokens_NFE_and_acceptance_required=True))
                if not parity:
                    raise RuntimeError(f'Same-R output/NFE/acceptance parity failed on request {index}; further sweep stopped')
        write(dest/'status.json',dict(success=True,R=a.R,mode=a.mode,backend=a.backend,
            gpu_mask=MASK,measurement_session=os.environ.get('POC_MULTIBD_SESSION','historical_gpu2367')))
    except BaseException as exc:
        write(dest/'status.json',dict(success=False,R=a.R,mode=a.mode,backend=a.backend,error=repr(exc)))
        raise
    finally:
        if llm is not None:llm.exit()


if __name__=='__main__':main()
