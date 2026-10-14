"""Request-level statistics. Last-replay estimates never become measured E2E."""
import argparse
import csv
import gzip
import statistics
import numpy as np
from poc_dllm_prefix_canvas_cp.contract import *
from poc_dllm_ep_comm.metrics import bootstrap_mean


def load(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def table(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)


def amdahl_reduction(factor,fraction):
    return fraction if factor=='infinite' else fraction*(1-1/float(factor))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',action='append');a=p.parse_args()
    directories=[ROOT/x for x in (a.input or ['results/raw/stage_a_native_v4','results/raw/stage_a_eager_prefill_v6'])]
    frozen=json.loads((ROOT/'results/frozen_manifest.json').read_text())
    requests=[];layers=[];steps=[];checks=[]
    for request in frozen['requests']:
        if request['split']!='calibration':continue
        rid=request['request_id']
        for prefix in PREFIXES:
            directory=next((d for d in reversed(directories) if list(d.glob(f'{rid}.p{prefix}.clean.0.rank0.json.gz'))),None)
            if directory is None:continue
            files=sorted(directory.glob(f'{rid}.p{prefix}.*.rank0.json.gz'))
            if not files:continue
            clean=[];marked=[];truth=None
            for file in files:
                rows=[load(Path(str(file).replace('.rank0.json.gz',f'.rank{rank}.json.gz'))) for rank in range(4)]
                if rows[0]['repeat']<0 or rows[0]['repeat']==99:continue
                if truth is None:truth=rows[0]
                for row in rows:
                    ok=(row['output_ids']==truth['output_ids'] and row['NFE']==truth['NFE']
                        and row['native_updates']==truth['native_updates'])
                    checks.append(dict(request_id=rid,prefix=prefix,rank=row['rank'],mode=row['mode'],parity=ok))
                    assert ok,(rid,prefix,row['rank'],row['mode'],'token/acceptance/NFE mismatch')
                if rows[0]['mode']=='clean':clean.append(rows)
                elif rows[0]['mode']=='coarse':marked.append(rows)
            if len(clean)!=3 or len(marked)!=2:continue
            critical=int(np.argmax([sum(run[rank]['wall_ms'] for run in marked) for rank in range(4)]))
            clean_critical=int(np.argmax([sum(run[rank]['wall_ms'] for run in clean) for rank in range(4)]))
            clean_wall=statistics.mean(max(r['wall_ms'] for r in run) for run in clean)
            event_wall=statistics.mean(max(r['wall_ms'] for r in run) for run in marked)
            native=clean[0][clean_critical]
            warm=statistics.mean(sum(x['ms'] for x in run[clean_critical]['phases'] if x['stage']=='decode') for run in clean)
            prefill=statistics.mean(sum(x['ms'] for x in run[clean_critical]['phases'] if x['stage']=='prefill') for run in clean)
            totals=[]
            for repeat,run in enumerate(marked):
                row=run[critical];call_by_id={x['call']:x for x in row['calls']}
                warm_attn=0;prefill_attn=0;moe=0
                for sample in row['graph_samples']:
                    key=sample['key'];count=sample['actual_native_calls'];is_warm=bool(key[1])
                    last_call=next(x for x in reversed(row['calls']) if
                        [1,x['cache_keys']>0,x['query_rows'],x['cache_keys']]==key)
                    for operator in sample['operators']:
                        ms=operator['ms']
                        layers.append(dict(request_id=rid,prefix=prefix,repeat=repeat,rank=critical,
                            query_rows=key[2],physical_keys=key[3],phase='warm' if is_warm else 'prefill',
                            native_calls=count,layer=operator['layer'],stage=operator['stage'],
                            last_replay_ms=ms,last_forward_ms=last_call['ms'],scope=sample['scope']))
                        if operator['stage']=='attention_path':
                            if is_warm:warm_attn+=count*ms
                            else:prefill_attn+=count*ms
                        if operator['stage']=='MLP_MoE':moe+=count*ms
                prep=sum(e['ms'] for e in row['events'] if e['stage']=='graph_input_KV_prepare'
                    and call_by_id[e['call']]['phase']!='prefill')
                event_warm=sum(x['ms'] for x in row['phases'] if x['stage']=='decode')
                totals.append(dict(estimated_attention_ms=warm_attn,measured_KV_prepare_ms=prep,
                    estimated_prefill_attention_ms=prefill_attn,
                    warm_target_superset_fraction=(warm_attn+prep)/event_warm,
                    cold_repeated_target_fraction=(warm_attn+prep)/row['wall_ms'],
                    cold_all_attention_fraction=(warm_attn+prep+prefill_attn)/row['wall_ms']))
                for call in row['calls']:
                    steps.append(dict(request_id=rid,prefix=prefix,rank=critical,repeat=repeat,**{
                        k:v for k,v in call.items() if k!='iteration'},iteration=json.dumps(call['iteration'])))
            result=dict(request_id=rid,domain=request['domain'],prefix=prefix,NFE=truth['NFE'],
                output_tokens=truth['output_tokens'],TPF=truth['output_tokens']/truth['NFE'],
                internal_first_update_ms=statistics.mean(run[clean_critical]['first_token_update_ms'] for run in clean),
                cold_E2E_ms=clean_wall,warm_decode_span_ms=warm,prefill_ms=prefill,
                cold_output_TPS=truth['output_tokens']/(clean_wall/1000),warm_output_TPS=truth['output_tokens']/(warm/1000),
                event_wall_ms=event_wall,observer_overhead=event_wall/clean_wall-1,
                cumulative_event_critical_rank=critical,clean_critical_rank=clean_critical,
                peak_HBM_allocated=max(r['max_memory_allocated'] for run in clean for r in run),
                peak_HBM_reserved=max(r['max_memory_reserved'] for run in clean for r in run),
                all_parity=True,KV_per_rank_bytes_proxy=20*2*1*1*(2*prefix)*128*2,
                attribution='Whole-attention superset; last-graph-replay stationarity estimate, NOT directly timed full-request attention')
            for field in totals[0]:result[field]=statistics.mean(x[field] for x in totals)
            requests.append(result)
    summaries={}
    for prefix in PREFIXES:
        selected=[r for r in requests if r['prefix']==prefix]
        if not selected:continue
        fields=[k for k,v in selected[0].items() if isinstance(v,(float,int)) and not isinstance(v,bool)
                and k not in ['prefix','cumulative_event_critical_rank','clean_critical_rank']]
        summary={k:bootstrap_mean([r[k] for r in selected]) for k in fields}
        summary['N']=len(selected)
        summary['event_status']='UNTRUSTED' if summary['observer_overhead']['ci95'][1]>.10 else 'VALIDATED_BOUNDED'
        summary['attribution_status']='STATIONARITY_ESTIMATE_ONLY; directly timed last replays + per-call graph KV prep'
        for scope in ['warm_target_superset_fraction','cold_repeated_target_fraction','cold_all_attention_fraction']:
            summary[scope+'_Amdahl']={str(s):bootstrap_mean([amdahl_reduction(s,r[scope]) for r in selected])
                                     for s in [1.5,2,4,'infinite']}
        summary['Amdahl_units']='Existing *_Amdahl fields are fractional LATENCY REDUCTION ceilings, not speedup factors'
        summary['Amdahl_speedup_bounds']={scope:{str(s):bootstrap_mean([
            1/(1-amdahl_reduction(s,r[scope])) for r in selected]) for s in [1.5,2,4,'infinite']}
            for scope in ['warm_target_superset_fraction','cold_repeated_target_fraction']}
        if prefix in [512,16384]:
            summary['prefill_attention_attribution']='UNKNOWN in primary coarse lane: native eager-prefill fallback; warm/repeated-target estimates unaffected'
            summary['cold_all_attention_fraction_status']='INCOMPLETE_DO_NOT_INTERPRET_AS_ALL_ATTENTION'
        summaries[str(prefix)]=summary
    table(ROOT/'results/baseline_per_request.csv',requests)
    table(ROOT/'results/baseline_per_step.csv',steps)
    table(ROOT/'results/baseline_per_layer.csv',layers)
    write(ROOT/'results/critical_path_summary.json',dict(prefixes=summaries,
        measurement_scope='Native physical TP4 graph baseline; all target fractions are generous stationarity estimates',
        last_replay_limitation='CUDA external events retain the last replay only. Earlier timestamps are not invented.',
        full_request_attention_directly_timed=False,raw_directories=[str(d.relative_to(ROOT)) for d in directories],
        heavy_profile_wall_is_headline=False,sample_unit='8 calibration requests; 2000 request bootstrap, not layers or repeats'))
    write(ROOT/'results/correctness.json',dict(checks=checks,all_parity=all(x['parity'] for x in checks),
        tested_comparisons=len(checks),scope='same-config native clean/coarse/trace/collectives output, acceptance updates and NFE',
        candidate='NOT_RUN',benchmark_accuracy='NOT_RUN; output64 is not complete task scoring'))
    print('Analyzed',len(requests),'request-length episodes; prefix counts',{k:v['N'] for k,v in summaries.items()})
    for prefix,v in summaries.items():print(prefix,'wall',round(v['cold_E2E_ms']['mean'],1),
        'observer%',round(v['observer_overhead']['mean']*100,2),'warm attention-upper%',
        round(v['warm_target_superset_fraction']['mean']*100,2),'cold repeated upper%',
        round(v['cold_repeated_target_fraction']['mean']*100,2),v['event_status'])


if __name__=='__main__':main()
