"""Aggregate completed evidence only. Resample independent examples, not edges."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

from .common import ROOT, SEED, write

def jsonl(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)

def table(name,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def estimate(values,weights=None):
    values=np.array(values,dtype=float)
    weights=np.ones(len(values)) if weights is None else np.array(weights,dtype=float)
    if not len(values):
        return dict(n=0,mean=None,ci95_lo=None,ci95_hi=None)
    rng=np.random.default_rng(SEED)
    indices=rng.integers(0,len(values),(4000,len(values)))
    boot=(values[indices]*weights[indices]).sum(1)/weights[indices].sum(1)
    return dict(n=len(values),mean=float(np.average(values,weights=weights)),
        ci95_lo=float(np.quantile(boot,.025)),ci95_hi=float(np.quantile(boot,.975)))

def quality_summary():
    groups=defaultdict(list)
    for p in (ROOT/'raw').glob('*_p2_*_shard*_v1/quality.jsonl'):
        if not (p.parent/'complete.json').exists():
            continue
        for r in jsonl(p):
            groups[(r['model'],r['split'],r['dataset'],r['arm'])].append(r)
    for p in (ROOT/'raw').glob('*_independent_control_shard*_v*/quality.jsonl'):
        if not (p.parent/'complete.json').exists():
            continue
        for r in jsonl(p):
            if r['arm'].startswith('N1_independent'):
                groups[(r['model'],r['split'],r['dataset'],r['arm'])].append(r)
    result=[]
    for (model,split,dataset,arm),rows in sorted(groups.items()):
        assert len({r['id'] for r in rows})==len(rows),('duplicate example',model,split,arm)
        ci=estimate([r['delta_nll'] for r in rows],[r['token_denominator'] for r in rows])
        result.append(dict(model=model,split=split,dataset=dataset,arm=arm,examples=len(rows),
            scored_tokens=sum(r['token_denominator'] for r in rows),
            actual_length_min=min(r['length'] for r in rows),actual_length_max=max(r['length'] for r in rows),
            baseline_nll=np.average([r['baseline_nll'] for r in rows],weights=[r['token_denominator'] for r in rows]),
            delta_nll=ci['mean'],delta_nll_ci95_lo=ci['ci95_lo'],delta_nll_ci95_hi=ci['ci95_hi'],
            delta_nll_worst=max(r['delta_nll'] for r in rows),
            kl_reference_to_variant=np.mean([r['kl_reference_to_variant'] for r in rows]),
            top1_agreement=np.mean([r['top1_agreement'] for r in rows]),
            effective_removed_fraction=np.mean([r.get('effective_removed_fraction',0) for r in rows]),
            effective_removed_fraction_work_weighted=sum(r.get('omitted_edges',0) for r in rows)/max(1,sum(r.get('total_edges',0) for r in rows)),
            total_edges=sum(r.get('total_edges',0) for r in rows) if arm!='B0' else None,
            executed_edges=sum(r.get('executed_edges',0) for r in rows) if arm!='B0' else None,
            omitted_edges=sum(r.get('omitted_edges',0) for r in rows),
            cache_peak_tensor_bytes=max((r['cache_peak_tensor_bytes'] for r in rows if 'cache_peak_tensor_bytes' in r),default=None),
            cache_accounting='LIVE_COUNTER' if all('cache_peak_tensor_bytes' in r for r in rows) else 'NOT_RECORDED_IN_PRIMARY_TEACHER_FORCING',
            mean_nll_gate=ci['mean']<=.02,upper_ci_nll_gate=ci['ci95_hi']<=.02,
            evidence='LIVE_FULL_MODEL_INTERVENTION',accuracy_equivalence='NOT_ESTABLISHED_BY_NLL'))
    table('quality_summary.csv',result)
    return result

def routing_summary():
    manifest=ROOT/'workload_manifest.json'
    if not manifest.exists():
        manifest=ROOT/'workload_manifest_public.json'
    workload={r['id']:r for r in json.loads(manifest.read_text())['rows']}
    aggregate=defaultdict(lambda:defaultdict(list))
    metadata={}
    counterexamples=[]
    paths=list((ROOT/'raw').glob('*_p1_*_v3/discovery*.jsonl'))
    paths+=list((ROOT/'raw').glob('*_p2_*_shard*_v1/discovery_from_p2.jsonl'))
    for p in paths:
        if not (p.parent/'complete.json').exists() or p.name=='discovery_generation.jsonl':
            continue
        for r in jsonl(p):
            if 'pairing' not in r:
                continue
            if r['pairing']=='shuffled_previous_pass' and r['pass_index']==1:
                continue
            split=workload[r['id']]['split']
            key=(r['model'],split,r['regime'],r['dataset'],r['pairing'],r['pass_index'])
            if 'counterexample' in r:
                counterexamples.append(dict(model=r['model'],id=r['id'],site=r['site'],pass_index=r['pass_index'],**r['counterexample']))
            for metric,value in r.items():
                if not isinstance(value,dict) or 'mean' not in value or value['mean'] is None:
                    continue
                mkey=key+(metric,)
                aggregate[mkey][r['id']].append((value['mean'],value['n'],value['p10'],value['p50'],value['p90']))
                metadata[mkey]=key
    route,representation=[],[]
    route_names={'jaccard','intersection_fraction','exact_set','top1','gate_weight_absolute_delta','selected_rank_changed'}
    for key,examples in sorted(aggregate.items()):
        model,split,regime,dataset,pairing,pass_index,metric=key
        means,weights,p10,p50,p90=[],[],[],[],[]
        for entries in examples.values():
            w=np.array([r[1] for r in entries])
            means.append(np.average([r[0] for r in entries],weights=w))
            weights.append(w.sum())
            p10.extend(r[2] for r in entries)
            p50.extend(r[3] for r in entries)
            p90.extend(r[4] for r in entries)
        ci=estimate(means,weights)
        row=dict(model=model,split=split,regime=regime,dataset=dataset,pairing=pairing,pass_index=pass_index,
            metric=metric,independent_examples=len(examples),sampled_observations=int(sum(weights)),
            mean=ci['mean'],ci95_lo=ci['ci95_lo'],ci95_hi=ci['ci95_hi'],
            mean_within_frame_p10=float(np.mean(p10)),mean_within_frame_p50=float(np.mean(p50)),mean_within_frame_p90=float(np.mean(p90)),
            quantile_scope='Average within-frame quantiles, not pooled token quantiles',evidence='LIVE_PROBE_DERIVED')
        (route if metric in route_names else representation).append(row)
    table('routing_summary.csv',route)
    table('representation_summary.csv',representation)
    counterexamples.sort(key=lambda r:r['output_relative_l2'],reverse=True)
    table('counterexamples.csv',counterexamples[:100])
    overall=defaultdict(lambda:defaultdict(list))
    for key,examples in aggregate.items():
        model,split,regime,dataset,pairing,pass_index,metric=key
        for example,entries in examples.items():
            overall[(model,split,regime,dataset,pairing,metric)][example].extend(entries)
    pooled=[]
    for key,examples in sorted(overall.items()):
        model,split,regime,dataset,pairing,metric=key
        values,weights=[],[]
        for entries in examples.values():
            w=[r[1] for r in entries]
            values.append(np.average([r[0] for r in entries],weights=w))
            weights.append(sum(w))
        pooled.append(dict(model=model,split=split,regime=regime,dataset=dataset,pairing=pairing,metric=metric,
            sampled_observations=sum(weights),**estimate(values,weights),evidence='LIVE_PROBE_DERIVED_SEQUENCE_BOOTSTRAP'))
    table('model_overall_summary.csv',pooled)
    return route,representation

def timing_summary():
    groups=defaultdict(list)
    micro=defaultdict(list)
    generation=[]
    for path in (ROOT/'raw').glob('*_timing_start*_v1'):
        if not (path/'complete.json').exists():
            continue
        request=json.loads((path/'request.json').read_text())
        for p in path.glob('timing_p*.json'):
            r=json.loads(p.read_text())
            events=defaultdict(float)
            for e in r['expert_events']:
                events[e['repeat']]+=e['expert_span_ms']
            groups[(r['model'],r['actual_length'])].append(dict(clean=np.mean(r['clean_wall_ms']),
                observer=np.mean(r['observer_wall_ms']),expert=np.mean(list(events.values())),
                gpu=request['gpu'],peak=r['peak_allocated_bytes']))
        p=path/'generation_timing.jsonl'
        if p.exists():
            generation.extend(dict(r,run=path.name,gpu=request['gpu']) for r in jsonl(p))
        p=path/'expert_microbench.jsonl'
        if p.exists() and not list((ROOT/'raw').glob('*_micro_v2_start*/complete.json')):
            for r in jsonl(p):
                micro[(r['model'],r['input_length'],r['target_q'])].append(r)
    for p in (ROOT/'raw').glob('*_micro_v2_start*/expert_microbench.jsonl'):
        if (p.parent/'complete.json').exists():
            for r in jsonl(p):
                micro[(r['model'],r['input_length'],r['target_q'])].append(r)
    result=[]
    for (model,length),values in sorted(groups.items()):
        ci=estimate([r['clean'] for r in values])
        expert=float(np.mean([r['expert'] for r in values]))
        observer=float(np.mean([r['observer'] for r in values]))
        ratio=observer/ci['mean']
        result.append(dict(model=model,input_length=length,independent_starts=len(values),micro_repeats_per_start=20,
            prefill_ms=ci['mean'],ci95_lo_ms=ci['ci95_lo'],ci95_hi_ms=ci['ci95_hi'],expert_event_span_ms=expert,
            observer_wall_ms=observer,observer_to_clean_ratio=ratio,
            expert_fraction_observer=expert/observer,expert_fraction_clean_optimistic=min(1.,expert/ci['mean']),
            fraction_validity='QUALIFIED_EVENT_SPAN' if abs(ratio-1)<=.1 else 'PARTIAL_OBSERVER_OVERHEAD',
            physical_gpus=','.join(sorted({r['gpu'] for r in values})),peak_allocated_bytes=max(r['peak'] for r in values),
            evidence='LIVE_MEASURED',cache='NONE_FULL_PREFIX_RECOMPUTATION'))
    table('timing_summary.csv',result)
    micro_rows=[]
    for (model,length,q),entries in sorted(micro.items()):
        ci=estimate([np.mean(r['wall_ms']) for r in entries])
        micro_rows.append(dict(model=model,input_length=length,target_q=q,starts=len(entries),
            effective_q=np.mean([r['effective_q'] for r in entries]),component_ms=ci['mean'],
            ci95_lo_ms=ci['ci95_lo'],ci95_hi_ms=ci['ci95_hi'],groups_nonempty=np.mean([r['groups_nonempty'] for r in entries]),
            evidence='LIVE_NATIVE_WEIGHT_COMPONENT_ONLY'))
    table('expert_microbench_summary.csv',micro_rows)
    table('baseline_generation_summary.csv',[dict(model=r['model'],run=r['run'],gpu=r['gpu'],
        prefix=r['actual_prefix'],generated_tokens=len(r['ids']),e2e_ms=r['e2e_s']*1000,ttft_ms=r['ttft_s']*1000,
        tpot_ms=r['tpot_s']*1000 if r['tpot_s'] is not None else None,
        output_tps=len(r['ids'])/r['e2e_s'],cache='NONE_FULL_PREFIX_RECOMPUTATION',evidence=r['evidence']) for r in generation])
    return result,micro_rows

def operational_summary():
    values=defaultdict(list)
    per_gpu=[]
    for p in (ROOT/'raw').glob('ifm_ouro_contention_*/contention.json'):
        if not (p.parent/'complete.json').exists():
            continue
        r=json.loads(p.read_text())
        mean=float(np.mean(r['wall_ms']))
        values[(r['participants'],int(r['physical_gpu']))].append(mean)
    for (participants,gpu),means in sorted(values.items()):
        per_gpu.append(dict(participants=participants,physical_gpu=gpu,starts=len(means),**estimate(means),
            evidence='LIVE_OPERATIONAL_CONTENTION_CONTROL'))
    table('contention_summary.csv',per_gpu)
    one=values.get((1,0),[])
    eight=values.get((8,0),[])
    write(ROOT/'contention_comparison.json',dict(physical_gpu=0,isolated_process_starts=len(one),
        parallel_process_starts=len(eight),parallel_to_isolated_ratio=float(np.mean(eight)/np.mean(one)) if one and eight else None,
        semantics='Independent single-GPU workers. Not model-parallel or serving throughput speedup.',
        other_devices='Owned burners during the single-worker condition. Independent model workers during the parallel condition.'))
    components=[]
    for p in (ROOT/'raw').glob('*_timing_start*_v1/components_p*.json'):
        if not (p.parent/'complete.json').exists():
            continue
        r=json.loads(p.read_text())
        categories=defaultdict(float)
        for event in r['events']:
            categories[event['category']]+=event['event_ms']
        for category,total in categories.items():
            components.append(dict(model=r['model'],run=p.parent.name,prefix=int(p.stem.split('p')[-1]),
                category=category,event_span_ms=total,observer_wall_ms=r['wall_ms'],
                evidence='SEPARATE_OBSERVER_MODULE_SPANS_NOT_CLEAN_100_PERCENT_PARTITION'))
    table('component_spans.csv',components)
    generation=[]
    for path in (ROOT/'raw').glob('*_heavy_profile_v1'):
        if not (path/'complete.json').exists():
            continue
        clean=json.loads((path/'generation_clean_control.json').read_text())
        profile=json.loads((path/'generation_expert_events.json').read_text())
        expert=sum(r['event_ms'] for r in profile['events'])
        generation.append(dict(model=profile['model'],prefix=128,tokens=16,clean_e2e_ms=clean['e2e_s']*1000,
            observer_e2e_ms=profile['observer_wall_s']*1000,expert_span_ms=expert,
            optimistic_fraction=expert/(clean['e2e_s']*1000),
            observer_fraction=expert/(profile['observer_wall_s']*1000),
            token_exact=profile['generated_ids']==clean['ids'],steps_measured=profile['steps'],
            evidence='LIVE_EACH_STEP_EVENTS_SEPARATE_CLEAN_GENERATION'))
    table('generation_fraction_summary.csv',generation)
    return per_gpu

def generation_summary():
    groups=defaultdict(dict)
    all_rows=[]
    for p in (ROOT/'raw').glob('*_generation_*/generation_quality.jsonl'):
        if not (p.parent/'complete.json').exists():
            continue
        for r in jsonl(p):
            kind='downstream_pilot' if r['cap']>=256 else 'short_rollout_fidelity'
            groups[(kind,r['model'],r['dataset'],r['arm'])][r['id']]=r
            all_rows.append(r)
    result=[]
    for (kind,model,dataset,arm),by_id in sorted(groups.items()):
        rows=list(by_id.values())
        reference=groups.get((kind,model,dataset,'B0'),{})
        pairs=[(r,reference[r['id']]) for r in rows if r['id'] in reference]
        ci=estimate([100*(int(x['correct'])-int(y['correct'])) for x,y in pairs])
        result.append(dict(scope=kind,model=model,dataset=dataset,arm=arm,examples=len(rows),
            correct=sum(r['correct'] for r in rows) if dataset!='language' else None,
            accuracy_percent=100*np.mean([r['correct'] for r in rows]) if dataset!='language' else None,
            accuracy_delta_pp=ci['mean'] if dataset!='language' else None,
            delta_ci95_lo_pp=ci['ci95_lo'] if dataset!='language' else None,
            delta_ci95_hi_pp=ci['ci95_hi'] if dataset!='language' else None,
            parsing_failures=sum(r['parsed_answer'] is None for r in rows),
            truncated_outputs=sum(r['truncated'] for r in rows),
            truncated_inputs=sum(r.get('input_truncated',False) for r in rows),
            eos=sum(r['finish_reason']=='eos' for r in rows),
            registered_stop=sum(r['finish_reason']=='registered_stop' for r in rows),
            token_exact_sequences=sum(r['token_exact'] for r in rows),
            output_tokens=sum(len(r['ids']) for r in rows),
            baseline_correct_variant_wrong=sum(y['correct'] and not x['correct'] for x,y in pairs),
            peak_expert_cache_tensor_bytes=max(r.get('cache_peak_tensor_bytes',0) for r in rows),
            evidence='LIVE_FREE_RUNNING_GREEDY',quality_equivalence='NOT_ESTABLISHED_UNDERPOWERED'))
    table('generation_quality_summary.csv',result)
    return result

def baseline_summary():
    result=[]
    for model,gpu in [('ifm_ouro',0),('ifm_huginn',3),('foil1',6)]:
        path=ROOT/'raw'/f'{model}_p0_gpu{gpu}_native_v5'
        if not (path/'complete.json').exists():
            continue
        meta=json.loads((path/'metadata.json').read_text())
        val=json.loads((path/'baseline_validation.json').read_text())
        result.append(dict(model=model,physical_gpu=gpu,parameter_count=meta['parameter_count'],
            persistent_state_elements=meta['persistent_state_elements'],registered_buffer_elements=meta['registered_buffer_elements'],
            physical_moe_sites=len(meta['physical_moe_sites']),recurrent_moe_sites=meta['recurrent_sites'],loops=meta['loops'],
            logical_moe_executions=val['total_site_executions'],native_dtype='BF16',cache='NONE',
            no_op_max_logit_error=val['no_op']['max_logit_error'],q0_max_logit_error=val['q0_max_error'],
            greedy_token_parity=val['exact_tokens'],backend=meta['backend'],load_seconds=meta['load_seconds'],
            parameter_bytes=meta['parameter_bytes'],allocated_bytes=meta['allocated_bytes'],
            baseline_status='PASS_NATIVE_NOOP_WITH_DOCUMENTED_BACKEND_DIFFERENCE' if model.startswith('ifm') else 'PASS'))
    table('baseline_summary.csv',result)
    return result

def confidence_summary():
    groups=defaultdict(lambda:defaultdict(list))
    for p in (ROOT/'raw').glob('*_p1_*_v3/confidence_contribution.jsonl'):
        if not (p.parent/'complete.json').exists():
            continue
        for r in jsonl(p):
            for metric in ['entropy','confidence','margin','relative_contribution','cancellation_ratio']:
                value=r[metric]
                if value['n']:
                    groups[(r['model'],r['dataset'],r['pass_index'],metric)][r['id']].append(value)
    result=[]
    for (model,dataset,loop,metric),examples in sorted(groups.items()):
        values,weights=[],[]
        for rows in examples.values():
            weights.append(sum(r['n'] for r in rows))
            values.append(np.average([r['mean'] for r in rows],weights=[r['n'] for r in rows]))
        result.append(dict(model=model,dataset=dataset,pass_index=loop,metric=metric,
            sampled_observations=sum(weights),**estimate(values,weights),
            average_frame_p10=np.mean([r['p10'] for rows in examples.values() for r in rows]),
            average_frame_p50=np.mean([r['p50'] for rows in examples.values() for r in rows]),
            average_frame_p90=np.mean([r['p90'] for rows in examples.values() for r in rows]),
            quantile_scope='Mean within-frame quantiles, not pooled token quantiles',
            evidence='LIVE_P1_SMOKE_PROBE_SEQUENCE_BOOTSTRAP'))
    table('confidence_contribution_summary.csv',result)
    return result

def main():
    baseline=baseline_summary()
    quality=quality_summary()
    routing,representation=routing_summary()
    timing,micro=timing_summary()
    generation=generation_summary()
    operational_summary()
    confidence_summary()
    bounds=[]
    for r in quality:
        matches=[t for t in timing if t['model']==r['model'] and t['input_length']==512]
        if r['split']!='heldout' or not matches:
            continue
        t=matches[0]
        q=r['effective_removed_fraction_work_weighted']
        bounds.append(dict(r,expert_fraction_observer=t['expert_fraction_observer'],
            proportional_ideal_speedup=1/(1-t['expert_fraction_observer']*q),
            clean_denominator_optimistic_speedup=1/(1-t['expert_fraction_clean_optimistic']*q),
            speedup_measured=None,speedup_evidence='ANALYTICAL_FLOP_PROPORTIONAL_AMDAHL_NOT_MEASURED',
            assumption='Saved expert time is proportional to omitted edges. Zero selection/cache overhead. This is not a proven universal bound.',
            timing_validity=t['fraction_validity']))
    table('oracle_summary.csv',[r for r in bounds if r['arm'].startswith(('O1','O2'))])
    table('candidate_summary.csv',[r for r in bounds if r['arm'].startswith(('C1','C2','N1'))])
    write(ROOT/'summary.json',dict(baseline=baseline,quality=quality,timing=timing,micro=micro,generation=generation,
        routing_rows=len(routing),representation_rows=len(representation),bounds=bounds,
        note='Only completed run directories enter these tables. Missing phases are not measured.'))
    print('SUMMARY',len(quality),len(timing),len(generation),len(bounds))

if __name__=='__main__':
    main()
