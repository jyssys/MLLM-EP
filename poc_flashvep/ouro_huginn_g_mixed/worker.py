"""An isolated pretrained worker. Heavy correctness and clean serving are separate."""
import argparse
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
import torch
from .common import ROOT, FORK, MODELS, DEPTHS, GROUPS, UPSTREAM, UUIDS, configuration, write, digest
from .policy import attach

def prompts(name, length, count):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[name], local_files_only=True, trust_remote_code=True)
    text = ('A public library has six shelves. Each shelf holds twelve books. '
        'Explain why careful counting matters. The garden receives rain and sunlight. ' * 4000)
    ids = tokenizer.encode(text, add_special_tokens=False)
    result = []
    for i in range(count):
        content = ids[i*17:i*17+length]
        assert len(content) == length
        result.append(content)
    return result

def load(name):
    from vllm_rlt.models import OuroForCausalLM, HuginnForCausalLM
    model = {'ouro': OuroForCausalLM, 'huginn': HuginnForCausalLM}[name].from_pretrained(
        MODELS[name], device='cuda:0', dtype=torch.bfloat16)
    if name == 'huginn':
        # Inference depth is an execution parameter. Original weights stay intact.
        model.config = replace(model.config, mean_recurrence=16)
    return model

def make_engine(model, job, arm, group, n=2):
    from vllm_rlt.engine.stage_engine import StageEngine
    cfg = configuration(job['model'], arm, group, job.get('C',512), n,
        job.get('graphs',True), job.get('async'))
    if job.get('blocks'):
        cfg['cache_config'] = replace(cfg['cache_config'], num_blocks=job['blocks'])
    engine = StageEngine(model, **cfg)
    policy = attach(engine, arm, group)
    return engine, policy

def parameters(name, cap, natural=True):
    from vllm_rlt import SamplingParams
    r = DEPTHS[name]
    return SamplingParams(max_tokens=cap, min_loops=r, max_loops=r,
        exit_threshold=1., temperature=0., ignore_eos=not natural, seed=20261025)

def destroy(engine):
    engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()

def difference(a, b):
    a,b = a.float(),b.float()
    return dict(exact=torch.equal(a,b), max_abs=float((a-b).abs().max()),
        rel_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)))

@torch.inference_mode()
def probe(model, job, arm, group):
    from vllm_rlt.request import Operation
    engine, policy = make_engine(model, job, arm, group)
    name, r = job['model'], DEPTHS[job['model']]
    ids = prompts(name, job['P'], 1)[0]
    engine.add_request('probe', ids, parameters(name, job.get('cap',8), False))
    runner, cm = engine.model_runner, engine.cache_manager
    native = runner.runtime.forward
    native_submit = runner.submit
    active = [None]
    tensor = {}
    trace = []
    def submit(scheduled_batch=None, *, batch):
        active[0] = batch
        trace.append(dict(operation=batch.operation.value, group=batch.loop_count,
            spans=[dict(id=s.request.request_id, count=s.token_count, start=s.token_start,
                depth=s.depth, chunk=s.chunk_start) for s in batch.spans]))
        return native_submit(scheduled_batch, batch=batch)
    def forward(batch):
        result = native(batch)
        if (batch.operation == Operation.CODA and batch.compute_logits
                and 'logits' not in tensor):
            tensor['hidden'] = batch.hidden[-1].detach().cpu().clone()
            tensor['logits'] = result.logits[-1].detach().cpu().clone()
            recurrent = set(getattr(model, 'recurrent_kv_layers', range(model.config.num_hidden_layers)))
            for layer in range(model.config.num_hidden_layers):
                depths = range(r) if layer in recurrent else [0]
                for depth in depths:
                    k,v = cm.read(layer, 'probe', depth, len(ids))
                    tensor[f'k_{depth}_{layer}'] = k.cpu().clone()
                    tensor[f'v_{depth}_{layer}'] = v.cpu().clone()
        return result
    runner.submit, runner.runtime.forward = submit, forward
    outputs = {}
    while engine.has_unfinished_requests():
        for output in engine.step(): outputs[output.request_id] = output
    runner.synchronize()
    runner.runtime.forward, runner.submit = native, native_submit
    output = outputs['probe']
    assert output.exit_depths == [r]*len(output.token_ids)
    result = dict(arm=arm, G=group, output=list(output.token_ids),
        depths=list(output.exit_depths), reason=output.finish_reason, trace=trace,
        cache_namespaces=(len(tensor)-2)//2, tensor_keys=sorted(tensor))
    # Remove closures before releasing the engine and its graph/cache storage.
    del forward, submit, native, native_submit, runner, cm, active, policy
    engine.close()
    del engine
    gc.collect();torch.cuda.empty_cache()
    return result,tensor

def support(model, job, target):
    reference = None
    rows = []
    arms = [('atomic',1),('split',1)]+[('group',g) for g in GROUPS[job['model']] if g>1]+[('mixed',1)]
    for arm,group in arms:
        print('SUPPORT',job['model'],arm,group,flush=True)
        row,tensor = probe(model,job,arm,group)
        if reference is None:
            reference = row,tensor
        else:
            ref, ref_tensor = reference
            assert tensor.keys() == ref_tensor.keys()
            metrics = {k:difference(ref_tensor[k],v) for k,v in tensor.items()}
            row['comparison'] = dict(tokens=row['output']==ref['output'],
                depths=row['depths']==ref['depths'], tensors=metrics,
                pass_tolerance=all(m['rel_l2']<=.03 and m['max_abs']<=.125 for m in metrics.values()))
        rows.append(row)
        write(target/f'{arm}_G{group}.json',row)
    write(target/'support.json',rows)
    return rows

def cohort(engine, job, arm, group, repeat, trace=False):
    from vllm_rlt.request import Operation
    n=job.get('N',4)
    name=job['model']
    params=parameters(name,job.get('cap',64),True)
    short=prompts(name,job.get('background_P',128),max(n,1))
    long=prompts(name,job.get('P',2048),1)[0]
    outputs,times,arrivals={}, {}, {}
    plans=[]
    runner=engine.model_runner
    graph_before={}
    for label,manager in [('core',runner.graphs),('coda',runner.coda_graphs)]:
        if manager is not None:
            graph_before[label]={k:getattr(manager,k) for k in ['captures','replays','fallbacks']}
    native=runner.submit
    def submit(scheduled_batch=None,*,batch):
        record=dict(host_s=time.perf_counter(),operation=batch.operation.value,
            group=batch.loop_count, rows=batch.num_rows,
            spans=[dict(id=s.request.request_id,depth=s.depth,start=s.token_start,
                count=s.token_count,prompt=s.chunk_start is not None) for s in batch.spans])
        if batch.operation==Operation.RECURRENT:
            mixed=batch.has_chunks and not batch.all_chunks
            assert not mixed or batch.loop_count==1
            record['mixed']=mixed
        plans.append(record)
        return native(scheduled_batch,batch=batch)
    if trace:runner.submit=submit
    started=time.perf_counter()
    def add(rid,ids):
        arrivals[rid]=time.perf_counter();times[rid]=[]
        engine.add_request(rid,ids,params)
    for i in range(n):add(f'b{i}',short[i])
    injected=False
    if n==0:add('long',long);injected=True
    deadline=time.monotonic()+job.get('cohort_timeout_s',600)
    while engine.has_unfinished_requests() or not injected:
        assert time.monotonic()<deadline,'Cohort timeout'
        if not injected and all(len(times[f'b{i}'])>=job.get('inject_after',4) for i in range(n)):
            add('long',long);injected=True
        updates=engine.step()
        for output in updates:
            now=time.perf_counter()
            old=outputs.get(output.request_id)
            old_count=len(old.token_ids) if old else 0
            if len(output.token_ids)>old_count:
                times[output.request_id].extend([now]*(len(output.token_ids)-old_count))
            outputs[output.request_id]=output
        # Background natural EOS before the trigger cannot hold the target forever.
        if not injected and not engine.has_unfinished_requests():
            add('long',long);injected=True
    engine.model_runner.synchronize()
    if trace:runner.submit=native
    end=time.perf_counter()
    records=[]
    for rid,out in outputs.items():
        assert out.exit_depths==[DEPTHS[name]]*len(out.token_ids)
        records.append(dict(id=rid,arrival=arrivals[rid]-started,
            tokens=list(out.token_ids),depths=list(out.exit_depths),
            times=[v-started for v in times[rid]],reason=out.finish_reason))
    gaps=[(b-a)*1000 for rid,ts in times.items() if rid!='long' for a,b in zip(ts,ts[1:])]
    ttft=(times['long'][0]-arrivals['long'])*1000
    graph_delta={}
    for label,manager in [('core',runner.graphs),('coda',runner.coda_graphs)]:
        if manager is not None:
            graph_delta[label]={k:getattr(manager,k)-v for k,v in graph_before[label].items()}
    return dict(model=name,arm=arm,G=group,repeat=repeat,P=job.get('P',2048),
        N=n,C=job.get('C',512),wall_s=end-started,TTFT_long_ms=ttft,
        ITL_p95_ms=float(np.percentile(gaps,95)) if gaps else None,
        ITL_p99_ms=float(np.percentile(gaps,99)) if gaps else None,
        output_tokens=sum(len(r['tokens']) for r in records),requests=records,plans=plans,
        peak_HBM=torch.cuda.max_memory_allocated(),graph_delta=graph_delta,
        cache_shape=list(engine.cache_manager.key_cache.shape),
        cache_pool_bytes=engine.cache_manager.key_cache.numel()*engine.cache_manager.key_cache.element_size()*2,
        cache_depths=engine.cache_manager.storage_depths,evidence='LIVE_MEASURED',
        lane='LIGHT_PLAN_OBSERVER' if trace else 'CLEAN_NATURAL_CAPPED_COHORT')

def screen(model, job, target):
    rows=[]
    arms=job.get('arms',[['atomic',1],['split',1],['mixed',1],['fair',1]]+
        [['group',g] for g in GROUPS[job['model']] if g>1]+
        [['hybrid',g] for g in GROUPS[job['model']] if g>1]+[['event',1]])
    offset=job.get('rotation',0)%len(arms);arms=arms[offset:]+arms[:offset]
    for arm,g in arms:
        engine,policy=make_engine(model,job,arm,g,job.get('N',4)+1)
        warm=dict(job)
        cohort(engine,warm,arm,g,-1)
        if hasattr(policy,'decisions'):policy.decisions=[]
        if job.get('barrier'):
            write(target/'READY.json',dict(wall_s=time.time()))
            release=ROOT/'state'/f"release_{job['barrier']}.json"
            deadline=time.monotonic()+1200
            while not release.exists():
                assert time.monotonic()<deadline
                time.sleep(.2)
            start=json.loads(release.read_text())['start_wall_s']
            while time.time()<start:time.sleep(.01)
        for repeat in range(job.get('repeats',2)):
            torch.cuda.reset_peak_memory_stats()
            row=cohort(engine,job,arm,g,repeat,job.get('trace',False))
            if hasattr(policy,'decisions'):
                row['selector_decisions']=policy.decisions
                policy.decisions=[]
            rows.append(row);write(target/f'{arm}_G{g}_r{repeat}.json',row)
            print('CELL',arm,g,repeat,round(row['TTFT_long_ms'],2),row['ITL_p99_ms'],flush=True)
        del policy
        engine.close();del engine;gc.collect();torch.cuda.empty_cache()
    write(target/'results.json',rows)
    return rows

def batched_audit(model,job,target):
    """Copy only valid produced values. These heavy runs are not timing evidence."""
    from vllm_rlt.request import Operation
    r=DEPTHS[job['model']]
    rows=[];reference=None
    for arm,g in job.get('arms',[['mixed',1],['hybrid',2],['hybrid',4],['group',2]]):
        engine,policy=make_engine(model,job,arm,g,job.get('N',2)+1)
        runner,cm=engine.model_runner,engine.cache_manager
        original_submit,original_forward=runner.submit,runner.runtime.forward
        active=[None];values={}
        def submit(scheduled_batch=None,*,batch):
            active[0]=batch
            return original_submit(scheduled_batch,batch=batch)
        def forward(batch):
            result=original_forward(batch)
            execution=active[0]
            if (batch.operation==Operation.CODA and batch.compute_logits and execution is not None):
                for index,span in enumerate(execution.spans):
                    rid=span.request.request_id
                    if rid!='long' or rid in values:continue
                    current=dict(hidden=batch.hidden[index].cpu().clone(),logits=result.logits[index].cpu().clone())
                    rec=set(getattr(model,'recurrent_kv_layers',range(model.config.num_hidden_layers)))
                    for layer in range(model.config.num_hidden_layers):
                        for depth in (range(r) if layer in rec else [0]):
                            k,v=cm.read(layer,rid,depth,job['P'])
                            current[f'k_{depth}_{layer}']=k[-16:].cpu().clone()
                            current[f'v_{depth}_{layer}']=v[-16:].cpu().clone()
                    values[rid]=current
            return result
        runner.submit,runner.runtime.forward=submit,forward
        row=cohort(engine,job,arm,g,0,False)
        assert set(values)=={'long'}
        if reference is None:reference=row,values
        else:
            ref,expected=reference
            metrics={k:difference(expected['long'][k],v) for k,v in values['long'].items()}
            row['numerical']=metrics
            row['tolerance_pass']=all(v['rel_l2']<=.03 and v['max_abs']<=.125 for v in metrics.values())
            row['target_output_exact']=next(x['tokens'] for x in row['requests'] if x['id']=='long')==next(x['tokens'] for x in ref['requests'] if x['id']=='long')
        row['lane']='HEAVY_NUMERICAL_NOT_PERFORMANCE'
        write(target/f'audit_{arm}_G{g}.json',row);rows.append({k:v for k,v in row.items() if k!='requests'})
        runner.submit,runner.runtime.forward=original_submit,original_forward
        del submit,forward,original_submit,original_forward,active,runner,cm,policy
        engine.close();del engine;gc.collect();torch.cuda.empty_cache()
    write(target/'batched_audit.json',rows)
    return rows

def dp_replica(model,job,target):
    """A real deterministic cluster burst. No throughput simulator is used."""
    from transformers import AutoTokenizer
    name=job['model'];arm=job['arm'];group=job.get('G',1)
    engine,policy=make_engine(model,job,arm,group,4)
    cohort(engine,dict(job,N=4,P=128,cap=12,inject_after=2),arm,group,-1)
    tokenizer=AutoTokenizer.from_pretrained(MODELS[name],local_files_only=True,trust_remote_code=True)
    cases=[]
    for index in job['global_ids']:
        text=tokenizer.apply_chat_template([dict(role='user',
            content=f'Calculate {index+3} plus 7. Reply with only the answer.')],
            tokenize=False,add_generation_prompt=True)
        cases.append((f'calc_{index}',tokenizer.encode(text,add_special_tokens=False)))
    write(target/'inputs.json',dict(cases=cases,placement='global_request_index_modulo_DP_width'))
    if job.get('chat_warmup'):
        warm_params=parameters(name,job.get('cap',128),True)
        warm_stop=tokenizer.convert_tokens_to_ids('<|im_end|>') if name=='ouro' else None
        warm_outputs={}
        for rid,ids in cases:engine.add_request(rid,ids,warm_params)
        while engine.has_unfinished_requests():
            for out in engine.step():
                warm_outputs[out.request_id]=list(out.token_ids)
                if warm_stop is not None and out.token_ids and out.token_ids[-1]==warm_stop:
                    engine.abort_request(out.request_id)
        engine.model_runner.synchronize()
        assert engine.cache_manager.num_used_blocks==0
        write(target/'matched_chat_warmup.json',dict(outputs=warm_outputs,
            qualifier='Completed outside the offered measurement window, with the same request IDs and actual prompts.'))
    graph_before={label:{k:getattr(manager,k) for k in ['captures','replays','fallbacks']}
        for label,manager in [('core',engine.model_runner.graphs),('coda',engine.model_runner.coda_graphs)] if manager is not None}
    write(target/'READY.json',dict(wall_s=time.time()))
    release=ROOT/'state'/f"release_{job['barrier']}.json"
    deadline=time.monotonic()+1200
    while not release.exists():
        assert time.monotonic()<deadline
        time.sleep(.2)
    start=json.loads(release.read_text())['start_wall_s']
    while time.time()<start:time.sleep(.001)
    times={rid:[] for rid,_ in cases};outputs={};reasons={}
    params=parameters(name,job.get('cap',128),True)
    for rid,ids in cases:engine.add_request(rid,ids,params)
    stop=tokenizer.convert_tokens_to_ids('<|im_end|>') if name=='ouro' else None
    while engine.has_unfinished_requests():
        for out in engine.step():
            now=time.time();rid=out.request_id
            old=outputs.get(rid)
            count=len(old.token_ids) if old else 0
            if len(out.token_ids)>count:times[rid].extend([now-start]*(len(out.token_ids)-count))
            outputs[rid]=out
            if stop is not None and out.token_ids and out.token_ids[-1]==stop:
                reasons[rid]='chat_stop'
                engine.abort_request(rid)
            elif out.finished:reasons[rid]=out.finish_reason
    engine.model_runner.synchronize()
    rows=[]
    for rid,out in outputs.items():
        assert out.exit_depths==[DEPTHS[name]]*len(out.token_ids)
        rows.append(dict(id=rid,tokens=out.token_ids,depths=out.exit_depths,
            reason=reasons[rid],times=times[rid],TTFT_s=times[rid][0],E2E_s=times[rid][-1]))
    result=dict(model=name,arm=arm,G=group,DP=job['width'],gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        cluster=job['barrier'],start_wall_s=start,finish_wall_s=time.time(),
        requests=rows,output_tokens=sum(len(r['tokens']) for r in rows),
        peak_HBM=torch.cuda.max_memory_allocated(),
        cache_shape=list(engine.cache_manager.key_cache.shape),
        cache_pool_bytes=engine.cache_manager.key_cache.numel()*engine.cache_manager.key_cache.element_size()*2,
        evidence='LIVE_MEASURED_DP_BURST',
        qualifier='One finite 32-request cluster burst. Not sustainable capacity or task accuracy.')
    result['condition']='MATCHED_CHAT_WARM' if job.get('chat_warmup') else 'LIBRARY_WARM_ONLY'
    result['graph_delta']={label:{k:getattr(manager,k)-v for k,v in graph_before[label].items()}
        for label,manager in [('core',engine.model_runner.graphs),('coda',engine.model_runner.coda_graphs)] if manager is not None}
    write(target/'dp_result.json',result)
    del policy;engine.close();del engine;gc.collect();torch.cuda.empty_cache()
    return result

def profile(model,job,target):
    rows=[]
    for arm,g in job['arms']:
        engine,policy=make_engine(model,job,arm,g,job.get('N',4)+1)
        cohort(engine,job,arm,g,-1)
        runner=engine.model_runner;original=runner.runtime.forward
        events=[]
        origin=torch.cuda.Event(enable_timing=True);origin.record()
        for stream in [runner.core_stream,runner.boundary_stream]:
            if stream is not None:stream.wait_event(origin)
        def forward(batch):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            before=time.perf_counter_ns();start.record()
            result=original(batch)
            end.record()
            events.append((dict(operation=batch.operation.value,
                recurrent_passes=len(batch.metadata) if hasattr(batch.metadata,'__len__') else None,
                rows=len(batch.hidden) if batch.hidden is not None else len(batch.tokens)
                    if batch.tokens is not None else None,
                graph_requested=batch.graphs is not None,
                host_interval_ns=time.perf_counter_ns()-before),start,end))
            return result
        runner.runtime.forward=forward
        row=cohort(engine,job,arm,g,0,True)
        runner.synchronize()
        row['event_intervals']=[dict(info,GPU_stream_interval_ms=a.elapsed_time(b),
            GPU_start_ms=origin.elapsed_time(a),GPU_end_ms=origin.elapsed_time(b)) for info,a,b in events]
        row['lane']='HEAVY_CUDA_EVENT_STREAM_INTERVALS_NOT_CLEAN_E2E'
        rows.append(row);write(target/f'profile_{arm}_G{g}.json',row)
        runner.runtime.forward=original
        del forward,original,runner,events,policy
        engine.close();del engine;gc.collect();torch.cuda.empty_cache()
    write(target/'profile.json',rows)
    return rows

def intervene(model,job,target):
    """One forced legal action after identical native prehistory, with state hashes."""
    import hashlib
    from vllm_rlt.request import RequestPhase as P
    rows=[]
    for action,g in [('mixed',1),('group',2),('group',4)]:
        engine,old_policy=make_engine(model,job,'mixed',1,job.get('N',1)+1)
        cohort(engine,job,'mixed',1,-1)
        native=engine.scheduler.policy
        info={}
        class Once:
            def __getattr__(self,key):return getattr(native,key)
            def record_batch(self,phase):native.record_batch(phase)
            def schedule(self,s,*,prefer_recurrent=False):
                target_request=s.requests.get('long')
                eligible=(not info and target_request is not None
                    and target_request.phase==P.PREFILLING_RECURRENT
                    and target_request.prefill_depth(target_request.num_prefilled_tokens)>=job.get('frontier',0)
                    and s.has_ready(P.PREFILLING_RECURRENT) and s.has_ready(P.RECURRENT))
                if eligible:
                    engine.model_runner.synchronize()
                    started=time.perf_counter()
                    state=[]
                    for rid,request in sorted(s.requests.items()):
                        if request.phase==P.FINISHED:continue
                        hidden=request.hidden_state
                        runner=engine.model_runner
                        if request.phase==P.PREFILLING_RECURRENT:
                            slot=runner.state_slots.get(rid)
                            if slot is not None:
                                hidden=runner.prefill_states[slot,:request.num_preluded_tokens-request.prefill_chunk_start]
                        elif hidden is None and runner.async_state is not None:
                            slot=runner.state_slots.get(rid)
                            if slot is not None:hidden=runner.states[slot]
                        record=dict(id=rid,position=request.position,depth=request.loops_done,
                            phase=request.phase.value,generated=list(request.generated_token_ids),
                            prefill_ranges=request.prefill_depth_ranges)
                        if hidden is not None:
                            record['hidden_sha']=hashlib.sha256(hidden.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                        samples=[]
                        allocation=engine.cache_manager._allocations.get(rid)
                        if allocation is not None:
                            for depth,writes in enumerate(allocation.written):
                                for layer,written in enumerate(writes):
                                    length=written.prefix
                                    if length:
                                        k,v=engine.cache_manager.read(layer,rid,depth,length)
                                        first=0 if job.get('full_state') else -1
                                        payload=torch.cat((k[first:].flatten(),v[first:].flatten())).cpu().contiguous().view(torch.uint8).numpy().tobytes()
                                        samples.append([depth,layer,length,hashlib.sha256(payload).hexdigest()])
                        record['valid_KV_full_hashes' if job.get('full_state') else 'valid_KV_tail_samples']=samples;state.append(record)
                    snapshot=json.dumps(state,sort_keys=True)
                    info.update(state_sha256=hashlib.sha256(snapshot.encode()).hexdigest(),
                        source_state=json.loads(snapshot),pause_for_state_hash_ms=(time.perf_counter()-started)*1000,
                        intervention_host_s=time.perf_counter())
                    if action=='group':
                        previous=s.config;s.config=replace(previous,prefill_loop_group_size=g)
                        try:return s._take(P.PREFILLING_RECURRENT)
                        finally:s.config=previous
                    return s.take_phases((P.RECURRENT,P.PREFILLING_RECURRENT))
                return native.schedule(s,prefer_recurrent=prefer_recurrent)
        engine.scheduler.policy=Once()
        row=cohort(engine,job,action,g,0,True)
        row['intervention']=info;row['lane']='HEAVY_VALID_STATE_ONE_ACTION_INTERVENTION'
        rows.append(row);write(target/f'intervention_{action}_G{g}.json',row)
        engine.scheduler.policy=native
        del native,old_policy,Once
        engine.close();del engine;gc.collect();torch.cuda.empty_cache()
    write(target/'interventions.json',rows)
    return rows

def observer(model,job,target):
    engine,policy=make_engine(model,job,'mixed',1,job.get('N',4)+1)
    cohort(engine,job,'mixed',1,-1)
    rows=[]
    modes=[False,True] if job.get('rotation',0)%2==0 else [True,False]
    for mode in modes:
        for rep in range(2):
            row=cohort(engine,job,'mixed',1,rep,mode)
            rows.append(row);write(target/f'observer_{mode}_r{rep}.json',row)
    write(target/'observer_comparison.json',rows)
    del policy;engine.close();del engine;gc.collect();torch.cuda.empty_cache()
    return rows

def lifecycle(model,job,target):
    name=job['model'];ids=prompts(name,64,2);rows=[]
    def finish(engine):
        outputs={}
        for _ in range(10000):
            if not engine.has_unfinished_requests():break
            for out in engine.step():outputs[out.request_id]=out
        else:raise AssertionError('Lifecycle run did not finish')
        return {rid:dict(tokens=o.token_ids,depths=o.exit_depths,reason=o.finish_reason) for rid,o in outputs.items()}
    for cancel in ['none','after_group','after_output']:
        engine,policy=make_engine(model,job,'hybrid',2,2)
        params=parameters(name,8,False)
        if cancel!='none':
            engine.add_request('reused',ids[0],params)
            if cancel=='after_group':
                engine.step();engine.step()
            else:
                while True:
                    out=engine.step()
                    if any(v.token_ids for v in out):break
            engine.abort_request('reused')
            engine.model_runner.synchronize()
            assert engine.cache_manager.num_used_blocks==0
        engine.add_request('reused',ids[1],params)
        engine.add_request('other',ids[0],params)
        output=finish(engine)
        assert engine.cache_manager.num_used_blocks==0
        assert all(v['depths']==[DEPTHS[name]]*len(v['tokens']) for v in output.values())
        rows.append(dict(cancel=cancel,outputs=output))
        del policy;engine.close();del engine;gc.collect();torch.cuda.empty_cache()
    assert all(v['outputs']==rows[0]['outputs'] for v in rows)
    write(target/'lifecycle.json',dict(cases=rows,reused_ID_parity=True,cache_released=True))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);a=p.parse_args()
    job=json.loads(Path(a.job).read_text())
    target=ROOT/'raw'/job['id'];target.mkdir(parents=True,exist_ok=True)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in [str(i) for i in range(8)]
    torch.set_num_threads(1)
    physical=int(os.environ['CUDA_VISIBLE_DEVICES'])
    uuid=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid==UUIDS[physical],(physical,uuid,UUIDS[physical])
    write(target/'identity.json',dict(job=job,pid=os.getpid(),start_wall=time.time(),
        GPU=os.environ['CUDA_VISIBLE_DEVICES'],gpu_uuid=uuid,model_revision=MODELS[job['model']].name,
        upstream=UPSTREAM,source_diff=subprocess.check_output(['git','diff'],cwd=FORK,text=True)))
    write(target/'harness_source_snapshot.json',{
        f.name:dict(sha256=digest(f),text=f.read_text())
        for f in Path(__file__).parent.glob('*.py')})
    try:
        if job['kind']!='dp':
            write(target/'input_token_ids.json',dict(
                target=prompts(job['model'],job.get('P',128),1)[0],
                background=prompts(job['model'],job.get('background_P',128),max(job.get('N',1),1)),
                source='Frozen library-counting mechanism text',question_benchmark=False))
        model=load(job['model'])
        write(target/'model_facts.json',dict(parameters=sum(p.numel() for p in model.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            config=model.config.to_dict(),fixed_R=DEPTHS[job['model']]))
        {'support':support,'screen':screen,'audit':batched_audit,'dp':dp_replica,
            'profile':profile,'intervene':intervene,'observer':observer,'lifecycle':lifecycle}[job['kind']](model,job,target)
        files=[dict(path=str(f.relative_to(target)),sha256=digest(f),bytes=f.stat().st_size)
            for f in target.rglob('*') if f.is_file() and f.name!='telemetry.jsonl']
        write(target/'completed.json',dict(end_wall=time.time(),artifacts=files))
    except Exception as error:
        write(target/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
