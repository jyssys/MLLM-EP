"""Measured discovery and full-model interventions. Oracles remain offline."""
from collections import defaultdict
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from .adapters import quality
from .common import ROOT, SEED, append, write
from .probes import Controller, oracle_plan
from .worker import timed

def similarity(a, b):
    a, b = a.float(), b.float()
    return dict(cosine=F.cosine_similarity(a,b,dim=-1),
        relative_l2=(a-b).norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-8),
        absolute_l2=(a-b).norm(dim=-1),
        norm_ratio=b.norm(dim=-1)/a.norm(dim=-1).clamp_min(1e-8))

def distribution(values):
    values = values.float().flatten()
    values = values[torch.isfinite(values)]
    if not len(values):
        return dict(n=0,mean=None,p10=None,p50=None,p90=None)
    quant = values.quantile(torch.tensor([.1,.5,.9])).tolist()
    return dict(n=len(values),mean=values.mean().item(),p10=quant[0],p50=quant[1],p90=quant[2])

def compare_frames(current, old, pairing, last_only=False, permutation=None):
    choose = slice(-1,None) if last_only else slice(None)
    ci, oi = current['ids'][choose], old['ids'][choose]
    cx, ox = current['x'][choose], old['x'][choose]
    co, oo = current['outputs'][choose], old['outputs'][choose]
    cw, ow = current['weights'][choose], old['weights'][choose]
    if permutation is not None:
        oi, ox, oo, ow = oi[permutation], ox[permutation], oo[permutation], ow[permutation]
    if pairing == 'spatial':
        adjacent = current['positions'][1:]-current['positions'][:-1] == 1
        ci,cx,co,cw = ci[1:][adjacent],cx[1:][adjacent],co[1:][adjacent],cw[1:][adjacent]
        oi,ox,oo,ow = oi[:-1][adjacent],ox[:-1][adjacent],oo[:-1][adjacent],ow[:-1][adjacent]
    match = ci[:,:,None] == oi[:,None,:]
    count = match.any(-1).sum(-1).float()
    k = ci.shape[-1]
    result = dict(pairing=pairing, positions=len(ci), matched_edges=int(match.sum()),
        jaccard=distribution(count/(2*k-count)), intersection_fraction=distribution(count/k),
        exact_set=distribution((count==k).float()), top1=distribution((ci[:,0]==oi[:,0]).float()))
    for name, value in similarity(cx,ox).items():
        result['normalized_input_'+name]=distribution(value)
    for name,value in similarity(current['raw_x'][choose],old['raw_x'][choose]).items() if pairing!='spatial' and permutation is None else []:
        result['raw_input_'+name]=distribution(value)
    rows, slots, oldslots = match.nonzero(as_tuple=True)
    result['gate_weight_absolute_delta']=distribution((cw[rows,slots]-ow[rows,oldslots]).abs())
    result['selected_rank_changed']=distribution((slots!=oldslots).float())
    for prefix, a, b in [('expert_output',co[rows,slots],oo[rows,oldslots]),
            ('weighted_output',co[rows,slots]*cw[rows,slots,None],oo[rows,oldslots]*ow[rows,oldslots,None])]:
        for name,value in similarity(a,b).items():
            result[prefix+'_'+name]=distribution(value)
    result['same_expert_input_relative_l2']=distribution(similarity(cx[rows],ox[rows])['relative_l2'])
    if len(rows) and pairing=='temporal_gap1':
        error=similarity(co[rows,slots],oo[rows,oldslots])['relative_l2']
        worst=int(error.argmax())
        positions=current['positions'][choose]
        result['counterexample']=dict(position=int(positions[rows[worst]]),expert=int(ci[rows[worst],slots[worst]]),
            output_relative_l2=float(error[worst]),
            output_absolute_l2=float((co[rows[worst],slots[worst]].float()-oo[rows[worst],oldslots[worst]].float()).norm()))
    return result

def frame_statistics(frames, context, last_only=False):
    histories = defaultdict(list)
    results=[]
    gen=torch.Generator().manual_seed(SEED)
    for frame in frames:
        if not frame['recurrent']:
            continue
        site, r = frame['site'], frame['pass_index']
        common=dict(context,site=site,pass_index=r,regime=context['regime'])
        for gap in [1,2,4,8]:
            if len(histories[site]) >= gap:
                row=compare_frames(frame,histories[site][-gap],f'temporal_gap{gap}',last_only)
                results.append(dict(common,**row))
        if histories[site] and not last_only:
            old=histories[site][-1]
            permutation=torch.randperm(len(frame['ids']),generator=gen)
            results.append(dict(common,**compare_frames(frame,old,'random_position_frequency_preserving',False,permutation)))
            shuffled=histories[site][int(torch.randint(len(histories[site]),(),generator=gen))]
            results.append(dict(common,**compare_frames(frame,shuffled,'shuffled_previous_pass',False)))
        if not last_only and len(frame['ids'])>1:
            # Only truly adjacent positions enter the spatial comparison.
            results.append(dict(common,**compare_frames(frame,frame,'spatial')))
        histories[site].append(frame)
    return results

def rows(job):
    manifest=json.loads((ROOT/'workload_manifest.json').read_text())
    selected=manifest['rows']
    if 'ids' in job:
        selected=[r for r in selected if r['id'] in set(job['ids'])]
    else:
        selected=[r for r in selected if r['dataset']=='language' and r['split']=='calibration'][:8]
    return selected

def discovery(a, target, job):
    count=0
    for row in rows(job):
        for length in job.get('lengths',[512]):
            ids=a.encode(row['text'],length)
            with Controller(a,'probe') as c:
                logits=a.forward(ids)
                frames=c.frames
            context=dict(model=a.name,id=row['id'],dataset=row['dataset'],length=ids.shape[-1],regime='prefill')
            for record in frame_statistics(frames,context):
                append(target/'discovery.jsonl',record)
            for f in frames:
                if f['recurrent']:
                    norm=f['outputs'].float().norm(dim=-1)*f['weights'].float()
                    contribution=norm/norm.sum(-1,keepdim=True).clamp_min(1e-8)
                    weighted=(f['outputs'].float()*f['weights'][...,None].float()).sum(1)
                    append(target/'confidence_contribution.jsonl',dict(context,site=f['site'],pass_index=f['pass_index'],
                        entropy=distribution(f['entropy']),confidence=distribution(f['confidence']),margin=distribution(f['margin']),
                        relative_contribution=distribution(contribution),
                        cancellation_ratio=distribution(weighted.norm(dim=-1)/norm.sum(-1).clamp_min(1e-8))))
            if count==0:
                # One small audit sample, not a dump of the model's activations.
                torch.save([{k:v for k,v in f.items() if k not in ('full_ids','full_contribution','full_reuse_error')}
                    for f in frames if f['site']==next(iter(a.recurrent))][:4],target/'audit_vectors.pt')
            count+=1
            if job.get('decode_steps',0):
                generated=[]
                for step in range(job['decode_steps']):
                    with Controller(a,'probe') as c:
                        logits=a.forward(ids)
                        context.update(regime='generation_last_position_full_prefix',step=step)
                        for record in frame_statistics(c.frames,context,last_only=True):
                            append(target/'discovery_decode.jsonl',record)
                    token=logits[:,-1].argmax(-1,keepdim=True)
                    generated.append(token.item())
                    ids=torch.cat([ids,token],dim=-1)
                append(target/'discovery_generation.jsonl',dict(id=row['id'],length=length,output_ids=generated,text=a.decode(generated)))
            print('P1_SEQUENCE',a.name,row['id'],length,flush=True)
    return dict(status='PASS',sequences=count,model=a.name)

def arms(include_independent=False):
    result=[dict(id='q0',mode='random_drop',q=0.)]
    for q in [.05,.1,.2,.3]:
        for renorm in [False,True]:
            result.append(dict(id=f'O1_drop_{q}_renorm{int(renorm)}',mode='oracle_drop',q=q,renorm=renorm))
        result.append(dict(id=f'O2_reuse_{q}',mode='oracle_reuse',q=q))
        result.append(dict(id=f'N1_random_drop_{q}',mode='random_drop',q=q))
    for threshold in [.05,.1,.2,.3]:
        result.append(dict(id=f'C1_router_{threshold}',mode='causal_drop',q=threshold))
    for guard in [0.,.05,.1,.2,.4,None]:
        result.append(dict(id=f'C2_reuse_guard{guard}',mode='reuse',q=1.,guard=guard))
    if include_independent:
        for q in [.05,.1,.2,.3]:
            result.append(dict(id=f'N1_independent_site_drop_{q}',mode='random_drop',q=q,random_scope='independent_site'))
    return result

def interventions(a,target,job):
    options=arms(include_independent=any(s.startswith('N1_independent') for s in job.get('arms',[])))
    if 'arms' in job:
        options=[r for r in options if r['id'] in job['arms']]
    count=0
    for row in rows(job):
        ids=a.encode(row['text'],job.get('length',128))
        reference=a.forward(ids)
        with Controller(a,'probe') as c:
            instrumented=a.forward(ids)
            assert torch.equal(reference,instrumented)
            plan=oracle_plan(c.frames)
            context=dict(model=a.name,id=row['id'],dataset=row['dataset'],length=ids.shape[-1],regime='prefill')
            for record in frame_statistics(c.frames,context):
                append(target/'discovery_from_p2.jsonl',record)
        append(target/'quality.jsonl',dict(model=a.name,id=row['id'],dataset=row['dataset'],split=row['split'],
            arm='B0',length=ids.shape[-1],**quality(reference,reference,ids)))
        order=options.copy()
        random.Random(SEED+count).shuffle(order)
        # Quality interventions are deterministic. Run matched controls after treatments.
        order.sort(key=lambda v: v['mode']=='random_drop')
        matched_budgets={}
        for spec in order:
            config={k:v for k,v in spec.items() if k!='id'}
            if spec['mode']=='random_drop' and spec['q'] in matched_budgets:
                config['matched_budget']=matched_budgets[spec['q']]
            with Controller(a,**config,oracle=plan) as c:
                variant=a.forward(ids)
                metrics=quality(variant,reference,ids)
                work=dict(c.work)
                assert work.get('total_edges',0)==work.get('executed_edges',0)+work.get('omitted_edges',0),work
                if spec['mode']=='oracle_drop' and not spec.get('renorm',False):
                    matched_budgets[spec['q']]=dict(c.edge_counts)
            if spec['id']=='q0':
                assert torch.equal(reference,variant), 'No-op intervention failed'
            append(target/'quality.jsonl',dict(model=a.name,id=row['id'],dataset=row['dataset'],split=row['split'],
                arm=spec['id'],length=ids.shape[-1],**metrics,**work,
                effective_removed_fraction=work.get('omitted_edges',0)/max(1,work.get('total_edges',0)),
                evidence='LIVE_FULL_MODEL_QUALITY_INTERVENTION_NOT_SPEEDUP',
                selection='OFFLINE_BASELINE_INFORMATION' if spec['mode'].startswith('oracle') else 'CAUSAL_CURRENT_OR_PRIOR_STATE'))
        count+=1
        print('P2_SEQUENCE',a.name,row['id'],count,flush=True)
    return dict(status='PASS',sequences=count,arms=len(options),model=a.name)

def timing(a,target,job):
    selected=rows(job)
    text=(selected[0]['text']+'\n')*32
    for length in job.get('lengths',[128,512,2048]):
        ids=a.encode(text,length)
        clean=timed(lambda:a.forward(ids),job.get('repeats',20),job.get('warmups',5))
        events=[]
        observer=[]
        for repeat in range(5):
            with Controller(a,'profile') as c:
                torch.cuda.synchronize()
                started=time.perf_counter()
                out=a.forward(ids)
                torch.cuda.synchronize()
                observer.append((time.perf_counter()-started)*1000)
                for site,r,start,end in c.times:
                    events.append(dict(repeat=repeat,site=site,pass_index=r,expert_span_ms=start.elapsed_time(end),
                        recurrent=site in a.recurrent))
        write(target/f'timing_p{length}.json',dict(model=a.name,requested_length=length,actual_length=ids.shape[-1],
            clean_wall_ms=clean,observer_wall_ms=observer,expert_events=events,
            evidence='LIVE_MEASURED', expert_span_scope='Sequential expert module CUDA-event spans. Not kernel-only service time.',
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),allocated_bytes=torch.cuda.memory_allocated()))
        from .timing import component_run, microbench
        component_run(a,ids,target,length)
        microbench(a,ids,target,length)
    for length in job.get('generation_prefix_lengths',[128,512]):
        ids=a.encode(text,length)
        for cap in job.get('generation_lengths',[16,64]):
            a.generate(ids,4,stop_eos=False)
            for repeat in range(job.get('generation_repeats',3)):
                result=a.generate(ids,cap,stop_eos=False)
                append(target/'generation_timing.jsonl',dict(model=a.name,requested_prefix=length,
                    actual_prefix=ids.shape[-1],cap=cap,repeat=repeat,**result,
                    evidence='LIVE_MEASURED_CAPPED_HARDWARE_GENERATION',stopping='Fixed cap, EOS disabled for hardware shape only. Not accuracy.'))
    from .timing import generation_components
    generation_components(a,a.encode(text,128),target,16)
    return dict(status='PASS',model=a.name)

def run(a,phase,target,job):
    write(target/'model_metadata.json',a.metadata())
    return {'p1':discovery,'p2':interventions,'timing':timing}[phase](a,target,job)
