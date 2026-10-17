"""Measured summaries and separately labeled architecture forecasts."""
import argparse
import collections
import csv
import statistics
import numpy as np
from poc_recurrent_asymmetry.contract import *
from poc_recurrent_asymmetry.dag import isomorphism,schedule


def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader();writer.writerows(rows)


def bootstrap(values):
    values=np.asarray(values,dtype=float)
    if len(values)<2:return None
    draws=np.random.default_rng(20261017).integers(0,len(values),(5000,len(values)))
    return np.quantile(values[draws].mean(1),[.025,.975]).tolist()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(exist_ok=False,parents=True)
    profiles=sorted((ROOT/'outputs').glob('profile_P*_v1'))
    samples=[];memory=[];skips=[];chunks=[]
    for folder in profiles:
        assert read(folder/'status.json')['success']
        samples += [read(f) for f in folder.glob('clean_B*.json')]
        skips += read(folder/'skipped.json')
        for f in folder.glob('chunk_*_repeat0.json'):
            value=read(f)
            if 'events' in value:chunks.append(value)
    grouped=collections.defaultdict(list)
    for row in samples:grouped[row['context'],row['batch']].append(row)
    summary=[]
    weights=read(ROOT/'outputs/baseline_v1/environment.json')['unique_weight_bytes']
    for (P,B),rows in sorted(grouped.items()):
        requests=collections.defaultdict(list)
        for row in rows:requests[tuple(row['request_ids'])].append(row)
        pref=[statistics.median(r['prefill_wall_ms'] for r in values) for values in requests.values()]
        dec=[statistics.median(r['decode_wall_ms']/r['decode_steps'] for r in values) for values in requests.values()]
        summary.append(dict(context=P,batch=B,independent_fixed_groups=len(requests),technical_samples=len(rows),
            prefill_ms=statistics.mean(pref),prefill_group_bootstrap_CI95=bootstrap(pref),
            cached_step_ms=statistics.mean(dec),cached_step_group_bootstrap_CI95=bootstrap(dec),
            fixed_step_output_tokens_per_s=B*1000/statistics.mean(dec),
            label='MEASURED_EXACT_HF_FIXED_STEP. NOT_NATURAL_OUTPUT_TPS.',
            sample_unit='Frozen request group. Repeated context lengths do not add independent questions.'))
        mem=rows[0]['prefill_memory'];logical=B*P*4*24*2*16*128*2
        assert mem['logical_KV_bytes']==logical and mem['slots']==96 and mem['key_storage_count']==96
        memory.append(dict(context=P,batch=B,unique_weight_GiB=weights/1024**3,
            logical_KV_GiB=logical/1024**3,allocated_KV_GiB=mem['allocated_KV_storage_bytes']/1024**3,
            peak_total_GiB=max(r['final_memory']['peak_allocated_bytes'] for r in rows)/1024**3,
            actual_HBM_read_write_bytes='UNAVAILABLE: ERR_NVGPUCTRPERM',
            live_append_copy_operand_bytes_per_step=2*logical,
            caveat='Append proxy counts old reads plus new writes. It is structural, not counter-measured HBM traffic.'))
    table(out/'profile_summary.csv',summary);write(out/'profile_summary.json',summary)
    table(out/'memory.csv',memory);write(out/'capacity_skips.json',skips)
    component=[]
    for source in (ROOT/'outputs/light_v1').glob('P*.json'):
        records=read(source)
        for kind in sorted({r['component'] for r in records}):
            selected=[r for r in records if r['component']==kind]
            divisor=8 if selected[0]['phase']=='decode' else 1
            observer=statistics.median(r['observer_percent'] for r in selected)
            clean=statistics.median(selected[0]['clean_wall_ms'])/divisor
            duration=statistics.median(r['component_sum_ms'] for r in selected)/divisor
            component.append(dict(P=selected[0]['P'],phase=selected[0]['phase'],component=kind,
                clean_ms_per_forward=clean,component_interval_ms=duration,
                interval_over_clean_percent=duration/clean*100,observer_percent=observer,
                timing_validity='VALID_SINGLE_FAMILY' if abs(observer)<=10 else 'UNVALIDATED',
                repeats=5,independent_requests=1,
                scope='Different component-family runs cannot be added as a measured exclusive E2E partition.'))
    table(out/'light_components.csv',component);write(out/'light_components.json',component)
    native=[];native_fidelity=[]
    for TP,foldername in [(1,'native_TP1_v1'),(2,'native_TP2_v1'),(4,'native_TP4_v3')]:
        folder=ROOT/'outputs'/foldername
        if not (folder/'status.json').exists():continue
        status=read(folder/'status.json');fidelity=read(folder/'fidelity.json')
        native_fidelity.append(dict(TP=TP,qualification=status['reference_qualification'],
            natural_prefixes_equal=sum(r['token_prefix_equal'] for r in fidelity['natural']),
            numerical_cases=fidelity['comparisons']))
        for P in [1024,4096,8192,16384]:
            rows=[read(f) for f in folder.glob(f'P{P}_*_repeat*.json')]
            group=collections.defaultdict(list)
            for r in rows:group[r['request_id']].append(r)
            for field in ['TTFT_ms','TPOT_ms','wall_ms']:
                values=[statistics.median(r[field] for r in group[k]) for k in sorted(group)]
                native.append(dict(TP=TP,P=P,metric=field,mean_request_median_ms=statistics.mean(values),
                    request_bootstrap_CI95=bootstrap(values),requests=len(group),technical_repeats=5,
                    reference_qualification=status['reference_qualification'],
                    timing_label='MEASURED_OPTIMIZED_NATIVE_CONTROL. Qualification remains explicit.'))
    table(out/'native_summary.csv',native);write(out/'native_summary.json',native);write(out/'native_fidelity.json',native_fidelity)
    pairs=[]
    for P in [1024,4096,8192,16384]:
        for field in ['TTFT_ms','TPOT_ms','wall_ms']:
            per_tp={}
            for TP,foldername in [(1,'native_TP1_v1'),(2,'native_TP2_v1'),(4,'native_TP4_v3')]:
                grouped_native=collections.defaultdict(list)
                for path in (ROOT/'outputs'/foldername).glob(f'P{P}_*_repeat*.json'):
                    value=read(path);grouped_native[value['request_id']].append(value[field])
                per_tp[TP]={k:statistics.median(v) for k,v in grouped_native.items()}
            for TP in [2,4]:
                shared=sorted(set(per_tp[1])&set(per_tp[TP]))
                if not shared:continue
                reductions=[100*(1-per_tp[TP][k]/per_tp[1][k]) for k in shared]
                pairs.append(dict(P=P,TP=TP,metric=field,request_pairs=len(shared),
                    mean_paired_reduction_percent=statistics.mean(reductions),CI95=bootstrap(reductions),
                    scope='Paired hardware diagnostic only. Cross-backend numerical qualification remains separate.'))
    write(out/'native_paired.json',pairs);table(out/'native_paired.csv',pairs)
    chunk_checks=read(ROOT/'outputs/chunk_correctness_v1/checks.json')
    write(out/'chunk_correctness_summary.json',dict(cases=len(chunk_checks),
        numerical_pass=sum(r['pass_numerical'] for r in chunk_checks),
        top1_pass=sum(r['top1_equal'] for r in chunk_checks),
        worst_logits_max_abs=max(r['logits_max_abs'] for r in chunk_checks),
        worst_logits_mean_abs=max(r['logits_mean_abs'] for r in chunk_checks),
        worst_KV_relative_L2=max(r['KV_relative_L2'] for r in chunk_checks),
        qualification='BOUNDED_PASS' if all(r['pass_numerical'] for r in chunk_checks) else 'REFERENCE_PARTIAL',
        scope='Full 96-slot KV, two calibration requests, prefill and three forced continuations. No pipeline engine.'))
    physical=ROOT/'outputs/physical_v1/physical'
    cp=read(physical/'cp_operator.json');p2p=read(physical/'p2p.json');collective=read(ROOT/'outputs/physical_v1/nccl/collectives.json')
    table(out/'CP_operator.csv',[dict(P=r['context'],loop=r['loop'],native_ms=r['native']['median_ms'],
        CP4_ms=r['CP4']['median_ms'],speedup=r['speedup'],correct=r['correctness'],
        max_abs=r['max_abs'],relative_L2=r['relative_L2'],KV_placement_ms=r['KV_placement_ms']) for r in cp])
    table(out/'p2p_summary.csv',[dict(source=r['source'],destination=r['destination'],bytes=r['payload_bytes'],
        host_ms=r['median_ms'],destination_stream_ms=statistics.median(r['destination_stream_ms']),
        apparent_GB_per_s=r['apparent_GB_per_s']) for r in p2p])
    oracle_a=[]
    for row in component:
        if row['phase']!='decode' or row['component']!='ATTN_CORE':continue
        P=row['P'];T=row['clean_ms_per_forward'];A=row['component_interval_ms']
        measured=[r for r in cp if r['context']==P]
        C=statistics.mean(r['CP4']['median_ms'] for r in measured)*96
        for scenario,new_attention in [('ZERO_COST_IDEAL',A/4),('MEASURED_OPERATOR_COST',C),
            ('CONSERVATIVE_OPERATOR_COST',max(r['CP4']['p95_ms'] for r in measured)*96)]:
            total=max(0.,T-A)+new_attention
            oracle_a.append(dict(P=P,scenario=scenario,reference_ms=T,attention_interval_ms=A,
                modeled_ms=total,modeled_reduction_percent=(1-total/T)*100,
                attention_infinite_speedup_ceiling_percent=A/T*100,
                attention_timing_validity=row['timing_validity'],
                label='MODELED_FROM_HF_AND_PHYSICAL_OPERATOR. NOT_MEASURED_E2E.',
                omitted_cost='Historical KV redistribution and fresh KV placement. Their nonnegative costs cannot rescue a negative result.',
                same_four_GPU_advantage='UNESTABLISHED. Standard CP operator only; native backend qualification is separate.',
                recurrent_specific_delta='No distinct recurrent mechanism was implemented or measured.'))
    write(out/'oracle_a.json',oracle_a)
    oracle_b=[];tapes=[]
    for row in chunks:
        tasks=collections.defaultdict(float)
        for event in row['events']:
            if event['category']=='LAYER':tasks[event['round'],event['layer']//6,event['chunk']]+=event['duration_ms']
            elif event['category']=='PASS_NORM':tasks[event['round'],3,event['chunk']]+=event['duration_ms']
        count=row['context']//row['chunk_rows'];assert len(tasks)==count*16
        message=row['chunk_rows']*2048*2
        comm=max(r['median_ms'] for r in p2p if r['payload_bytes']==message)
        residual=max(0.,statistics.median(row['clean_wall_ms'])-sum(tasks.values()))
        for scenario,quantum,cost_scale,conservative in [('ZERO_COST_IDEAL',0.,1.,False),
            ('MEASURED_COST',comm,1.,False),('CONSERVATIVE',comm*2,1.10,True)]:
            costs={k:v*cost_scale for k,v in tasks.items()}
            for placement,owner in [('layer_stationary_wrap',lambda t:t[1]),('loop_stationary_replicated',lambda t:t[0])]:
                candidates=[]
                for policy in ['chunk_fifo','asap']:
                    duration,timeline=schedule(costs,owner,quantum,policy,conservative)
                    candidates.append((duration,policy,timeline))
                duration,policy,timeline=min(candidates,key=lambda r:r[0])
                baseline=next(r for r in summary if r['context']==row['context'] and r['batch']==1)
                oracle_b.append(dict(P=row['context'],chunk=row['chunk_rows'],placement=placement,scenario=scenario,
                    selected_generic_policy=policy,modeled_TTFT_ms=duration+residual,
                    HF_single_GPU_prefill_ms=baseline['prefill_ms'],
                    modeled_reduction_vs_HF_single_percent=(1-(duration+residual)/baseline['prefill_ms'])*100,
                    measured_chunk_observer_percent=row['observer_percent'],
                    timing_validity='VALID_COMPONENT' if abs(row['observer_percent'])<=10 else 'UNVALIDATED_COMPONENT',
                    numerical_scope='Last-token equality measured. Full chunk KV/continuation probe is REFERENCE_PARTIAL. Conditional schedule only.',
                    recurrent_specific_gain_over_same_generic_schedule_percent=0.,
                    model='Exact coarse DAG. Full previous-chunk task completion is a conservative KV publication boundary.',
                    limitation='HF stage costs, not an optimized PP engine. No measured E2E speedup or free decode/KV transition claim.'))
                if row['context']==8192 and row['chunk_rows']==1024 and scenario=='MEASURED_COST':
                    tapes.extend(dict(P=8192,chunk=1024,placement=placement,scenario=scenario,**r) for r in timeline)
    table(out/'oracle_b.csv',oracle_b);write(out/'oracle_b.json',oracle_b);table(out/'modeled_timeline.csv',tapes)
    checks=[dict(chunks=c,**isomorphism(c)) for c in [1,2,4,8,16,32,64]]
    write(out/'dag_checks.json',checks)
    attention_ceiling=max((r['attention_infinite_speedup_ceiling_percent'] for r in oracle_a),default=None)
    validated_attention=bool(oracle_a) and all(r['attention_timing_validity']=='VALID_SINGLE_FAMILY' for r in oracle_a)
    operator_slower=all(r['speedup']<1 for r in cp)
    track_a=('A_NO_BOTTLENECK' if validated_attention and attention_ceiling<10 else
             'A_COMM_DOMINATED' if validated_attention and operator_slower else 'A_INCONCLUSIVE')
    generic_equivalence=all(r['edge_sets_equal'] for r in checks)
    track_b='B_GENERIC_ONLY' if generic_equivalence else 'B_INCONCLUSIVE'
    overall='GENERIC_ONLY' if generic_equivalence and all(r['correctness'] for r in cp) else 'INCONCLUSIVE'
    write(out/'summary.json',dict(clean_samples=len(samples),physical_P2P_cases=len(p2p),CP_cases=len(cp),
        independent_primary_question_IDs=8,primary_exact_reference='PASS',
        hardware_counters='UNAVAILABLE: permission denied. No system setting changed.',
        track_a=track_a,attention_infinite_ceiling_percent=attention_ceiling,
        attention_single_family_validated=validated_attention,CP_operator_slower=operator_slower,
        track_a_novelty='Only conventional CP is tested. No separate recurrent-specific mechanism.',
        track_b=track_b,generic_edge_isomorphism=generic_equivalence,
        overall=overall,architecture_GO=False,
        strong_exact_four_GPU_performance_gate='Not established by partial native cross-backend fidelity.',
        caveat='This does not prove that every future recurrent architecture lacks headroom. No candidate inference engine was built.'))
    print('ANALYSIS',len(summary),'shapes',len(oracle_a),'A models',len(oracle_b),'B models',len(native),'native summary rows')


if __name__=='__main__':main()
