"""Separate clean wall time from observer spans and structural byte accounting."""
from collections import defaultdict
import json
import numpy as np
from .common import ROOT,write
from .aggregate_quality import csv_write

def ci(values):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(20261026)
    boot=rng.choice(a,(4000,len(a)),replace=True).mean(1)
    return dict(mean=float(a.mean()),p50=float(np.median(a)),p90=float(np.quantile(a,.9)),
        ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),n=len(a))

def cv(values):
    a=np.asarray(values,dtype=float)
    return float(a.std()/max(a.mean(),1e-12))

def gini(values):
    a=np.sort(np.asarray(values,dtype=float));n=len(a)
    return float(np.sum((2*np.arange(1,n+1)-n-1)*a)/(n*max(a.sum(),1e-12)))

def aggregate():
    baseline=[];stages=[];imbalance=[];memory=[];generic=[];lead_times=[]
    for model in ['ifm_ouro','ifm_huginn','foil1']:
        for ep in [4,8]:
            directory=ROOT/'raw'/f'{model}_ep{ep}_full_profile_v1'
            for b in [1,2,4,8,16]:
                for length in [128,512,2048]:
                    key=dict(model=model,ep=ep,batch=b,prefix_tokens=length)
                    forward=directory/f'forward_b{b}_p{length}.rank0.json'
                    generation=directory/f'generation_b{b}_p{length}.rank0.jsonl'
                    if not forward.exists() or not generation.exists():
                        status='PENDING' if not (directory/'failure.rank0.json').exists() else 'FAILED'
                        baseline.append(dict(key,status=status,evidence='NOT_MEASURED'));continue
                    data=json.loads(forward.read_text());gen=[json.loads(s) for s in generation.read_text().splitlines()]
                    if len(gen)<3:
                        baseline.append(dict(key,status='IN_PROGRESS',completed_generation_repetitions=len(gen),evidence='NOT_COMPLETE'))
                        continue
                    row=dict(key,status='PASS',evidence='LIVE_MEASURED_TRUE_EP',output_cap=32,
                        decode_policy='Full-prefix recomputation, no cached decode',output_eos=False,
                        ci_scope='Serial warmed repetitions in one process start. No restart CI.')
                    for prefix,values in [('prefill_ms',data['clean_ms']),('e2e_ms',[v['e2e_ms'] for v in gen]),
                            ('ttft_ms',[v['ttft_ms'] for v in gen]),('tpot_ms',[v['tpot_ms'] for v in gen])]:
                        row.update({prefix+'_'+k:v for k,v in ci(values).items()})
                    all_generations=[]
                    for rank in range(ep):
                        p=directory/f'generation_b{b}_p{length}.rank{rank}.jsonl'
                        if p.exists():all_generations.append([json.loads(s) for s in p.read_text().splitlines()])
                    if len(all_generations)!=ep or min(map(len,all_generations))<3:
                        baseline.append(dict(key,status='IN_PROGRESS',evidence='RANK_OUTPUTS_NOT_COMPLETE'))
                        continue
                    makespans=[max(g[i]['e2e_ms'] for g in all_generations) for i in range(3)]
                    row.update({'cohort_makespan_ms_'+k:v for k,v in ci(makespans).items()})
                    row['e2e_endpoint']='Rank 0 output endpoint. Cohort makespan uses the maximum rank duration.'
                    row['output_tokens_s']=b*32*1000/row['cohort_makespan_ms_mean']
                    row['requests_s']=b*1000/row['cohort_makespan_ms_mean']
                    row['observer_ep_ratio']=data['observer_ep_ms']/row['prefill_ms_mean']
                    row['observer_router_ratio']=data['observer_router_ms']/row['prefill_ms_mean']
                    baseline.append(row)
                    per_rank=[]
                    for rank in range(ep):
                        p=directory/f'forward_b{b}_p{length}.rank{rank}.json'
                        if p.exists():
                            m=json.loads(p.read_text())
                            ownership=json.loads((directory/f'ownership.rank{rank}.json').read_text())
                            metadata=json.loads((directory/f'metadata.rank{rank}.json').read_text())
                            local_expert=sum(t['bytes'] for site in ownership for t in site['tensors'])
                            total_expert=sum(site['per_expert_bytes']*site['logical_experts'] for site in ownership)
                            memory.append(dict(key,rank=rank,peak_allocated_bytes=m['peak_allocated_bytes'],
                                allocated_bytes=m['allocated_bytes'],reserved_bytes=m['reserved_bytes'],
                                resident_expert_weight_bytes=local_expert,
                                resident_dense_weight_bytes=metadata['parameter_bytes']-total_expert,
                                registered_buffer_bytes=metadata['registered_buffer_bytes'],
                                peak_scope='Clean forward. Not full-generation peak or NVML total HBM.',
                                reserve_scope='Allocator snapshot after separate observer forwards.',
                                kv_cache_scope='No retained KV. Native paths recompute full-prefix attention.',
                                evidence='LIVE_CUDA_ALLOCATOR_AND_EXACT_SHARDED_STORAGE_ACCOUNTING'))
                    for phase,name in [('prefill','ep_trace'),('decode_last_forward','ep_decode')]:
                        observations=[]
                        for rank in range(ep):
                            p=directory/f'{name}_b{b}_p{length}.rank{rank}.json'
                            if p.exists():observations.append(json.loads(p.read_text()))
                        if len(observations)!=ep:continue
                        previous={(v['site'],v['loop']):v for v in observations[0] if v['recurrent']}
                        for (site,loop),current in previous.items():
                            nxt=previous.get((site,loop+1))
                            if nxt is not None:
                                lead_times.append(dict(key,phase=phase,site=site,previous_loop=loop,
                                    complete_to_next_ready_ms=nxt['ready_forward_ms']-current['complete_forward_ms'],
                                    ready_to_next_ready_ms=nxt['ready_forward_ms']-current['ready_forward_ms'],
                                    evidence='SAME_RANK_CUDA_EVENT_OBSERVER_INTERVAL',
                                    qualification='Potential preparation interval only. Exact next-loop activation does not yet exist.'))
                        # One rank supplies the complete non-overlapping decomposition.
                        rank=max(range(ep),key=lambda r:sum(v['total_span_ms'] for v in observations[r]))
                        spans=observations[rank]
                        out=dict(key,phase=phase,rank=rank,observer_ms=data['observer_ep_ms'] if phase=='prefill' else '',
                            evidence='LIVE_CUDA_EVENT_API_SPANS_NOT_EXCLUSIVE_KERNEL_TIME',
                            qualification='One cumulative-max rank. Decode row is one final forward, not all decode steps.')
                        for component in ['metadata_pack','dispatch','receive_grouping','expert','combine','restore','total_span']:
                            out[component+'_ms']=sum(v[component+'_ms'] for v in spans)
                        out['dispatch_remote_payload_bytes']=sum(v['remote_bytes'] for group in observations for v in group)
                        out['combine_remote_payload_bytes']=out['dispatch_remote_payload_bytes']
                        width=1024 if model=='foil1' else 1536
                        call_tokens=sum(v['batch']*v['length'] for v in spans)
                        out['remote_dispatch_edge_fraction']=out['dispatch_remote_payload_bytes']/max(call_tokens*2*width*2,1)
                        out['dispatch_nonempty_remote_peer_sends']=sum(sum(n>0 and peer!=rank for peer,n in enumerate(v['send_splits']))
                            for rank,group in enumerate(observations) for v in group)
                        out['replicated_restore_logical_bytes_per_rank']=sum(v['restoration_bytes'] for v in spans)
                        out['collective_count_per_rank']=len(spans)*4
                        out['all_to_all_count_per_rank']=len(spans)*2
                        stages.append(out)
                        for index,call in enumerate(spans):
                            loads=np.array(call['rank_loads'],dtype=float)
                            times=[observations[r][index]['expert_ms'] for r in range(ep)]
                            spans_time=[observations[r][index]['total_span_ms'] for r in range(ep)]
                            imbalance.append(dict(key,phase=phase,site=call['site'],loop=call['loop'],recurrent=call['recurrent'],
                                expert_load_cv=cv(call['expert_counts']),rank_load_cv=cv(loads),
                                expert_load_gini=gini(call['expert_counts']),rank_load_gini=gini(loads),
                                active_expert_ranks=int(np.count_nonzero(loads)),participating_ep_ranks=ep,
                                rank_load_max_mean=float(loads.max()/max(loads.mean(),1e-12)),
                                expert_event_span_cv=cv(times),expert_event_max_minus_mean_ms=float(max(times)-np.mean(times)),
                                ep_span_cv=cv(spans_time),
                                warning='Max-minus-mean span is dispersion, not measured stream idle or exact rank-wait time.',
                                evidence='LIVE_EVENT_SPANS_AND_EXACT_ROUTE_COUNTS'))
                    path=directory/f'generic_generation_b{b}_p{length}.rank0.jsonl'
                    fp=directory/f'generic_forward_b{b}_p{length}.rank0.json'
                    if path.exists() and fp.exists():
                        values=[json.loads(s) for s in path.read_text().splitlines()];f=json.loads(fp.read_text())
                        if len(values)<3:continue
                        g=dict(key,evidence='LIVE_GENERIC_CONTROL',source='Current top-1 expert owner among replicated hidden-state copies',
                            generic_e2e_ms=float(np.mean([v['e2e_ms'] for v in values])),
                            generic_prefill_ms=float(np.mean(f['clean_ms'])),
                            token_exact_requests=all(v['static_token_exact'] for v in values),
                            resident_state_movement_bytes=0,
                            qualification='Dense states already replicated. Not free owner migration with attention/KV sharding.')
                        g['static_to_generic_e2e_speedup']=row['e2e_ms_mean']/g['generic_e2e_ms']
                        g['static_to_generic_prefill_speedup']=row['prefill_ms_mean']/g['generic_prefill_ms']
                        traces=[]
                        for rank in range(ep):
                            p=directory/f'generic_trace_b{b}_p{length}.rank{rank}.json'
                            if p.exists():traces.append(json.loads(p.read_text()))
                        if len(traces)==ep:
                            g['remote_dispatch_payload_bytes']=sum(v['remote_bytes'] for rank in traces for v in rank)
                            g['observer_ms']=f['observer_ms']
                            critical=max(traces,key=lambda r:sum(v['total_span_ms'] for v in r))
                            for component in ['metadata_pack','dispatch','receive_grouping','expert','combine','restore']:
                                g['observer_'+component+'_ms']=sum(v[component+'_ms'] for v in critical)
                            # Static full-source matrices reconstruct every rank's exact payload.
                            static_bytes=0
                            for rank in range(ep):
                                p=directory/f'ep_trace_b{b}_p{length}.rank{rank}.json'
                                static_bytes+=sum(v['remote_bytes'] for v in json.loads(p.read_text()))
                            g['remote_dispatch_payload_reduction_fraction']=1-g['remote_dispatch_payload_bytes']/max(static_bytes,1)
                        generic.append(g)
    csv_write(ROOT/'baseline_e2e.csv',baseline)
    csv_write(ROOT/'baseline_stage_breakdown.csv',stages)
    csv_write(ROOT/'rank_imbalance.csv',imbalance)
    csv_write(ROOT/'memory_per_rank.csv',memory)
    csv_write(ROOT/'generic_control.csv',generic)
    csv_write(ROOT/'routing_lead_time.csv',lead_times)
    csv_write(ROOT/'batch_scaling.csv',baseline)
    write(ROOT/'runs/completeness.json',dict(required_cells=90,complete_cells=sum(r.get('status')=='PASS' for r in baseline),
        staged_rows=len(stages),generic_live_cells=len(generic)))
    return baseline,stages,imbalance,generic

if __name__=='__main__':
    b,s,i,g=aggregate();print('COMPLETE',sum(r['status']=='PASS' for r in b),'of',len(b),'GENERIC',len(g))
