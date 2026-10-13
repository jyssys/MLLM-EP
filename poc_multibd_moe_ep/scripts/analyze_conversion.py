"""Strict result accounting and request bootstrap; no speculative data fills."""
import collections
import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path
import numpy as np
from poc_multibd_moe_ep.contract import *
from poc_dllm_ep_comm.metrics import traffic,graph_stats,bootstrap_mean


def request_metrics(result):
    t=result['trajectories'][0]
    steps=t['trajectory']
    runtime_committed=sum(s['num_generated_tokens'] for s in steps)
    committed=len(t['token_ids'])
    wall_ms=result['wall_ms']
    key=(result['request']['domain'],result['request']['selected_index'])
    return dict(request_id=str(key),R=result['R'],mode=result['mode'],backend=result['backend'],
        measurement_session=result.get('measurement_session','historical_gpu2367'),
        gpu_mask=result.get('gpu_mask','2,3,6,7'),
        deterministic_ops=result.get('deterministic_ops',False),
        split=result['request']['split'],domain=result['request']['domain'],repeat=result['repeat'],
        nfe=len(steps),newly_committed_runtime=runtime_committed,useful_committed=committed,
        returned_tokens=len(t['token_ids']),wall_ms=wall_ms,
        TPF=committed/len(steps) if steps else None,
        TPS=1000*committed/wall_ms if wall_ms else None,
        step_ms=wall_ms/len(steps) if steps else None,
        truncated=t['max_new_tokens_reached'] or t['max_nfe_reached'],
        completion_reason=t['completion_reason'])


def ep_metrics(directory):
    traces=[]
    for r in range(4):
        p=directory/f'trace.events.rank{r}.json.gz'
        if p.exists():
            with gzip.open(p,'rt') as inp:traces.append(json.load(inp))
        else:traces.append(json.loads((directory/f'trace.events.rank{r}.json').read_text()))
    groups=collections.defaultdict(list)
    for rank,trace in enumerate(traces):
        for row in trace['records']:
            groups[(row['forward_index'],row['layer'])].append(row)
    cumulative=[sum(x['dispatch_ms']+x['expert_ms']+x['combine_ms']
                    for x in trace['records']) for trace in traces]
    critical_rank=int(np.argmax(cumulative))
    rows=[]
    for (forward,layer),records in sorted(groups.items()):
        assert len(records)==4
        loads=np.zeros(4,dtype=int);edges=np.zeros((4,4),dtype=int)
        union=set();d=c=remote=0
        expected_rows=np.zeros(4,dtype=int);observed_rows=np.zeros(4,dtype=int)
        observed_paths=np.zeros(4,dtype=int)
        for row in records:
            source=row['source_rank'];ids=np.asarray(row['ids'],dtype=int)
            accounting=traffic(ids,source,4096)
            loads+=np.bincount(accounting['owners'].ravel(),minlength=4)
            edges[source]+=accounting['destinations'].sum(0)
            expected_rows+=accounting['destinations'].sum(0)
            observed_rows[source]=row['recv_rows'];observed_paths[source]=row['recv_paths']
            d+=int(accounting['dispatch'].sum());c+=int(accounting['combine'].sum())
            remote+=int(accounting['remote_paths'].sum());union.update(ids.ravel().tolist())
        assert np.array_equal(expected_rows,observed_rows)
        assert np.array_equal(loads,observed_paths)
        critical=next(x for x in records if x['source_rank']==critical_rank)
        expert_counts=np.bincount([expert//64 for expert in union],minlength=4)
        # One real rank's components, never unrelated maxima added together.
        row=dict(forward=forward,layer=layer,dispatch_bytes=d,combine_bytes=c,
            remote_paths=remote,active_expert_union=len(union),
            active_experts_per_rank=expert_counts.tolist(),
            max_rank_active_experts=int(expert_counts.max()),
            tokens_per_active_expert=int(loads.sum())/len(union) if union else 0,
            destination_fanout=float(edges.sum()/sum(r['rows'] for r in records)),
            rank_assignments=loads.tolist(),critical_rank=critical['source_rank'],
            D_ms=critical['dispatch_ms'],E_ms=critical['expert_ms'],C_ms=critical['combine_ms'])
        row.update(graph_stats(edges,loads));rows.append(row)
    return rows


def paired_conversion(rows):
    repetitions=collections.defaultdict(list)
    for row in rows:
        if row['mode']=='clean':
            repetitions[(row['request_id'],row['backend'],row['R'])].append(row)
    grouped=collections.defaultdict(dict)
    for (request,backend,r),values in repetitions.items():
        # Repeats are noise observations of ONE problem, not independent samples.
        grouped[(request,backend)][r]={key:statistics.mean(v[key] for v in values)
            for key in ['TPF','TPS','step_ms']}
    results=[]
    for (request,backend),values in grouped.items():
        if 1 not in values:continue
        base=values[1]
        for r,current in sorted(values.items()):
            tg=current['TPF']/base['TPF'];sg=current['TPS']/base['TPS']
            results.append(dict(request_id=request,backend=backend,R=r,
                TPF_gain=tg,TPS_gain=sg,PCE=sg/tg,
                Delta_step_ms=current['step_ms']-base['step_ms']))
    return results


def load_events(directory,mode):
    traces=[]
    for rank in range(4):
        with gzip.open(directory/f'{mode}.events.rank{rank}.json.gz','rt') as inp:
            traces.append(json.load(inp))
    groups=collections.defaultdict(list)
    for trace in traces:
        for row in trace['records']:
            groups[(row['forward_index'],row['layer'])].append(row)
    totals=[sum(x['dispatch_ms']+x['expert_ms']+x['combine_ms'] for x in t['records'])
            for t in traces]
    critical_rank=int(np.argmax(totals))
    return traces,groups,critical_rank


def event_metrics(directory,mode):
    traces,groups,critical_rank=load_events(directory,mode)
    rows=[]
    for (forward,layer),records in sorted(groups.items()):
        assert len(records)==4
        critical=next(r for r in records if r['source_rank']==critical_rank)
        loads=[next(r for r in records if r['source_rank']==rank)['recv_paths'] for rank in range(4)]
        rows.append(dict(forward=forward,layer=layer,critical_rank=critical_rank,
            D_ms=critical['dispatch_ms'],E_ms=critical['expert_ms'],C_ms=critical['combine_ms'],
            whole_MoE_ms=critical['moe_total_ms'],rank_assignments=loads,
            physical_rows=sum(r['rows'] for r in records),
            max_rank_load=max(loads),rank_max_mean=max(loads)/np.mean(loads),
            rank_cv=float(np.std(loads)/np.mean(loads))))
    return rows


def summarize_ep(rows,committed,nfe,structural=False):
    result={key:sum(x[key] for x in rows) for key in ['D_ms','E_ms','C_ms']}
    for key in ['D','E','C']:
        result[key+'_ms_per_forward']=result[key+'_ms']/nfe
    result['EP_ms']=result['D_ms']+result['E_ms']+result['C_ms']
    result['EP_ms_per_commit']=result['EP_ms']/committed
    result['EP_ms_per_forward']=result['EP_ms']/nfe
    result['whole_MoE_ms']=sum(x.get('whole_MoE_ms',0) for x in rows)
    for key in ['rank_max_mean','rank_cv','max_rank_load']:
        result[key]=float(np.mean([r[key] for r in rows]))
    if structural:
        result['logical_D_C_bytes']=sum(x['dispatch_bytes']+x['combine_bytes'] for x in rows)
        result['D_C_bytes_per_commit']=result['logical_D_C_bytes']/committed
        for key in ['active_expert_union','tokens_per_active_expert','destination_fanout',
                    'edge_density','remote_edge_density','active_destination_ranks',
                    'max_rank_active_experts']:
            result[key]=float(np.mean([r[key] for r in rows]))
        result['active_experts_per_rank']=[float(np.mean([r['active_experts_per_rank'][rank]
                                                        for r in rows])) for rank in range(4)]
        # Actual owner tensors are 6 MiB per complete layer-specific expert.
        # This is an active-weight working-set proxy, NOT measured HBM reads.
        result['max_rank_active_weight_bytes_proxy']=result['max_rank_active_experts']*6*1024**2
    return result


def acceptance_signature(result):
    t=result['trajectories'][0]
    return dict(tokens=t['token_ids'],reason=t['completion_reason'],steps=[
        {key:step[key] for key in ['block_trace','buffer_bids','is_prefill',
                                  'num_generated_tokens','running_token_ids','step_id']}
        for step in t['trajectory']])


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    columns=sorted(set().union(*(r.keys() for r in rows)))
    # Complete layer records can exceed GitHub's individual-file limit.
    # This is lossless CSV compression, not trace/sample reduction.
    compressed=path.name=='ep_metrics.csv'
    target=path.with_suffix('.csv.gz') if compressed else path
    with (gzip.open(target,'wt',newline='') if compressed else target.open('w',newline='')) as out:
        writer=csv.DictWriter(out,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True)
    args=parser.parse_args()
    results=ROOT/'results'/args.session
    rows=[];loaded=[];layers=[];forwards=[];parity=[];invalid=[]
    for p in sorted((ROOT/'artifacts').glob('R*/request*.json')):
        result=json.loads(p.read_text())
        # Never pool interrupted old-GPU held-out samples with new hardware.
        if result['request']['split']=='held_out' and result.get('measurement_session')!=args.session:
            continue
        if result['bounded_smoke'] or result.get('is_warmup'):continue
        row=request_metrics(result);row['artifact']=str(p.relative_to(ROOT));loaded.append((row,result,p))
        if row['backend']=='ep4' and row['mode'] in ['timing','trace']:
            eventdir=p.with_suffix('')
            inputs=None
            try:eventrows=event_metrics(eventdir,row['mode'])
            except (AssertionError,FileNotFoundError) as exc:
                invalid.append(dict(artifact=row['artifact'],error=repr(exc),
                    label='INCOMPLETE_OR_CROSS_RANK_TRACE_MISMATCH',preserved=True))
                rows.append(row);continue
            if row['mode']=='trace':
                inputs=load_events(eventdir,row['mode'])[0][0]['forward_inputs']
                # Structural truth is captured from real collectives, but heavy
                # trace CUDA times are NOT used for headline stage attribution.
                structural=ep_metrics(eventdir)
                assert len(structural)==len(eventrows)
                for a,b in zip(structural,eventrows):
                    assert (a['forward'],a['layer'])==(b['forward'],b['layer'])
                    a['whole_MoE_ms']=b['whole_MoE_ms']
                eventrows=structural
            row.update(summarize_ep(eventrows,row['useful_committed'],row['nfe'],row['mode']=='trace'))
            perforward=collections.defaultdict(list)
            for event in eventrows:
                event.update(request_id=row['request_id'],R=row['R'],split=row['split'],
                    mode=row['mode'],repeat=row['repeat']);layers.append(event)
                perforward[event['forward']].append(event)
            assert len(perforward)==row['nfe'],(p,len(perforward),row['nfe'])
            for f,eventlist in sorted(perforward.items()):
                assert len(eventlist)==19
                step=result['trajectories'][0]['trajectory'][f]
                blocks=step['block_trace']
                forward=dict(request_id=row['request_id'],R=row['R'],mode=row['mode'],
                    split=row['split'],repeat=row['repeat'],forward=f,is_prefill=step['is_prefill'],
                    runtime_newly_committed=step['num_generated_tokens'],
                    active_blocks_after_step=sum(b['is_active'] and not b['is_dummy'] for b in blocks),
                    live_masks_after_step=sum(b['num_mask_tokens'] for b in blocks if not b['is_dummy']),
                    state_timestamp='POST_SAMPLER native trajectory; not pre-forward state',
                    D_ms=sum(x['D_ms'] for x in eventlist),
                    E_ms=sum(x['E_ms'] for x in eventlist),C_ms=sum(x['C_ms'] for x in eventlist))
                if row['mode']=='trace':
                    forward.update(logical_D_C_bytes=sum(x['dispatch_bytes']+x['combine_bytes'] for x in eventlist),
                        active_expert_union=float(np.mean([x['active_expert_union'] for x in eventlist])),
                        destination_fanout=float(np.mean([x['destination_fanout'] for x in eventlist])))
                    if inputs[f].get('before_forward_state'):
                        states=inputs[f]['before_forward_state']
                        forward.update(active_blocks_before=sum(s['active_blocks'] for s in states),
                            live_masks_before=sum(s['live_mask_tokens'] for s in states),
                            slots_before=states,physical_rows=inputs[f]['num_tokens'])
                        if not step['is_prefill']:
                            slots=sum((s['slots'] for s in states),[])
                            forward['dummy_blocks_before']=sum(s['is_dummy'] for s in slots)
                            forward['to_cache_blocks_before']=sum('TO_CACHE' in s['status'] for s in slots)
                            forward['dummy_row_fraction']=forward['dummy_blocks_before']*32/inputs[f]['num_tokens']
                forwards.append(forward)
        rows.append(row)
    grouped=collections.defaultdict(list)
    for row,result,p in loaded:grouped[(row['split'],row['request_id'],row['backend'],row['R'])].append((row,result))
    averaged=[]
    for key,values in grouped.items():
        split,request,backend,r=key
        clean=[v for v in values if v[0]['mode']=='clean']
        if not clean:continue
        reference=acceptance_signature(clean[0][1])
        tests=[dict(mode=row['mode'],repeat=row['repeat'],pass_all=acceptance_signature(res)==reference)
               for row,res in values]
        parity.append(dict(split=split,request_id=request,backend=backend,R=r,
            comparisons=tests,pass_all=all(t['pass_all'] for t in tests)))
        base=dict(clean[0][0]);base['repeats']=len(clean)
        states=[x for x in forwards if x['request_id']==request and x['R']==r and
                x['split']==split and x['mode']=='trace' and not x['is_prefill']] if backend=='ep4' else []
        if states:
            base['maximum_actual_active_blocks']=max(x.get('active_blocks_before',0) for x in states)
            base['mean_actual_active_blocks']=float(np.mean([x.get('active_blocks_before',0) for x in states]))
            base['mean_dummy_row_fraction']=float(np.mean([x.get('dummy_row_fraction',0) for x in states]))
        for metric in ['wall_ms','step_ms','TPS','TPF']:
            base[metric]=statistics.mean(v[0][metric] for v in clean)
        for mode in ['timing','trace']:
            observations=[v[0] for v in values if v[0]['mode']==mode and 'D_ms' in v[0]]
            if not observations:continue
            for metric in ['D_ms','E_ms','C_ms','EP_ms','whole_MoE_ms','EP_ms_per_commit','EP_ms_per_forward',
                           'D_ms_per_forward','E_ms_per_forward','C_ms_per_forward',
                           'logical_D_C_bytes','D_C_bytes_per_commit','active_expert_union','tokens_per_active_expert',
                           'max_rank_active_experts','max_rank_active_weight_bytes_proxy',
                           'destination_fanout','edge_density','rank_max_mean','rank_cv','max_rank_load']:
                if metric in observations[0]:
                    base[mode+'_'+metric]=statistics.mean(v[metric] for v in observations)
            base[mode+'_observer_wall_ratio']=statistics.mean(v['wall_ms'] for v in observations)/base['wall_ms']
        averaged.append(base)
    conversion=[]
    for split in ['calibration','held_out']:
        converted=paired_conversion([r for r in averaged if r['split']==split])
        lookup={(r['request_id'],r['backend'],r['R']):r for r in averaged if r['split']==split}
        for row in converted:
            row['split']=split
            current=lookup[(row['request_id'],row['backend'],row['R'])]
            baseline=lookup[(row['request_id'],row['backend'],1)]
            for metric in ['timing_EP_ms_per_commit','trace_D_C_bytes_per_commit','trace_active_expert_union',
                           'trace_tokens_per_active_expert','trace_destination_fanout',
                           'trace_rank_max_mean','trace_max_rank_load']:
                if metric in current and metric in baseline:
                    row[metric+'_ratio']=current[metric]/baseline[metric]
            if 'timing_EP_ms_per_forward' in current and 'timing_EP_ms_per_forward' in baseline:
                row['Delta_EP_ms']=current['timing_EP_ms_per_forward']-baseline['timing_EP_ms_per_forward']
                row['EP_incremental_share']=row['Delta_EP_ms']/row['Delta_step_ms'] if row['Delta_step_ms']>0 else None
            conversion.append(row)
    summary={}
    metrics=['TPF_gain','TPS_gain','PCE','Delta_step_ms','EP_incremental_share',
             'timing_EP_ms_per_commit_ratio','trace_D_C_bytes_per_commit_ratio',
             'trace_active_expert_union_ratio','trace_rank_max_mean_ratio']
    for split in ['calibration','held_out']:
        for backend in ['ep4','tp4']:
            summary[split+'_'+backend]={str(r):{key:bootstrap_mean([v[key] for v in conversion
                if v['R']==r and v['split']==split and v['backend']==backend and v.get(key) is not None])
                for key in metrics} for r in [1,2,3,4]}
    write_csv(results/'raw_repeats.csv',rows)
    write_csv(results/'per_request.csv',averaged)
    write_csv(results/'per_forward.csv',forwards)
    write_csv(results/'ep_metrics.csv',layers)
    write(results/'validity.json',dict(instrumentation_parity=parity,
        pass_all=all(v['pass_all'] for v in parity),
        held_out_pass_all=all(v['pass_all'] for v in parity if v['split']=='held_out'),
        ep4_held_out_pass_all=all(v['pass_all'] for v in parity
                               if v['split']=='held_out' and v['backend']=='ep4'),
        tp4_held_out_pass_all=all(v['pass_all'] for v in parity
                               if v['split']=='held_out' and v['backend']=='tp4'),
        invalid_traces=invalid,
        event_observer_rule='same-session held-out: <=5% preferred; >10% UNTRUSTED stage attribution',
        calibration_separate_sessions='variation confounded; not observer overhead proof'))
    write(results/'conversion_summary.json',dict(requests=averaged,paired=conversion,summary=summary,
        measurement_session=args.session,hardware_strata_not_pooled=True,
        sample_unit='request, not layer or refinement',bootstrap_replicates=2000))
    print('Main non-smoke request results:',len(rows))


if __name__=='__main__':main()
