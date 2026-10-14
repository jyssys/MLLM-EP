"""Physical baseline statistics separated from non-timing structural oracles."""
import collections
import csv
import gzip
import json
import statistics
import numpy as np
from poc_looped_moe_ep.contract import *
from poc_looped_moe_ep.metrics import traffic,jsd,balanced_candidate
from poc_dllm_ep_comm.metrics import bootstrap_mean


def gz(path):
    with gzip.open(path,'rt') as inp:return json.load(inp)


def table(path,rows):
    with gzip.open(path,'wt',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)


def episode(directory,key,placement,mapping,tag):
    run_records=[];clean=[];event=[];parity=[]
    for path in sorted(directory.glob(f'{tag}_*.rank0.json')):
        info=json.loads(path.read_text())
        four=[json.loads(Path(str(path).replace('.rank0.json',f'.rank{r}.json')).read_text()) for r in range(4)]
        assert all(row['pass_all'] for row in four)
        parity.extend(four)
        wall=max(row['wall_ms'] for row in four)
        if info['mode']=='clean':clean.append(wall)
        elif info['mode']=='timing':
            event.append(wall)
            run_records.append([gz(Path(str(path).replace('.rank0.json',f'.rank{r}.events.json.gz'))) for r in range(4)])
    assert len(clean)==len(event)==3
    cumulative=[sum(row['dispatch_ms']+row['expert_ms']+row['combine_ms']+row['restoration_ms']
        for repeat in run_records for row in repeat[rank]) for rank in range(4)]
    critical=int(np.argmax(cumulative))
    trace=[gz(directory/f'{tag}_trace_6.rank{r}.events.json.gz') for r in range(4)]
    assert all(row['native_router_ids_equal'] and row['native_router_weights_equal'] and
               row['native_residual_sample_equal'] for records in trace for row in records)
    for rank in range(4):
        assert json.loads((directory/f'{tag}.rank{rank}.greedy.json').read_text())['pass_all']
    row=dict(request_id=key,placement=placement,clean_ms=statistics.mean(clean),
        event_ms=statistics.mean(event),observer_overhead=statistics.mean(event)/statistics.mean(clean)-1,
        cumulative_critical_rank=critical,exact_logit_comparisons=sum(e['exact'] for item in parity for e in item['logit_errors']),
        total_logit_comparisons=sum(len(item['logit_errors']) for item in parity),
        pass_all=True,greedy_parity=True)
    for field in ['dispatch_ms','expert_ms','combine_ms','restoration_ms']:
        row[field]=statistics.mean(sum(x[field] for x in repeat[critical]) for repeat in run_records)
    row['sparse_EP_ms']=row['dispatch_ms']+row['expert_ms']+row['combine_ms']
    row['sparse_EP_plus_restore_ms']=row['sparse_EP_ms']+row['restoration_ms']
    row['event_EP_share']=row['sparse_EP_ms']/row['event_ms']
    metrics=[];histograms=[];first_forward_ids=[]
    for forward in range(3):
        for loop in range(16):
            source_rows=[next(x for x in rank if x['pool']=='core' and x['forward']==forward and x['loop']==loop)
                         for rank in trace]
            accounting=[traffic(x['ids'],r,mapping) for r,x in enumerate(source_rows)]
            counts=sum((a['expert_counts'] for a in accounting),start=np.zeros(64,dtype=int))
            loads=sum((a['rank_counts'] for a in accounting),start=np.zeros(4,dtype=int))
            expected_rows=sum((a['destinations'].sum(0) for a in accounting),start=np.zeros(4,dtype=int))
            assert np.array_equal(expected_rows,[r['recv_rows'] for r in source_rows])
            assert np.array_equal(loads,[r['recv_paths'] for r in source_rows])
            edges=np.stack([a['destinations'].sum(0) for a in accounting])
            active=[int(sum(counts[e]>0 and mapping[e]==rank for e in range(64))) for rank in range(4)]
            times=[]
            for rank in range(4):
                times.append({field:statistics.mean(next(x[field] for x in repeat[rank]
                    if x['pool']=='core' and x['forward']==forward and x['loop']==loop) for repeat in run_records)
                    for field in ['dispatch_ms','expert_ms','combine_ms','restoration_ms']})
            candidate=balanced_candidate(counts,mapping)
            oracle_load=[int(sum(counts[e] for e in range(64) if candidate[e]==rank)) for rank in range(4)]
            ids_all=np.empty((sum(x['rows'] for x in source_rows),2),dtype=int)
            for rank,x in enumerate(source_rows):ids_all[rank::4]=x['ids']
            if forward==0:
                histograms.append(counts);first_forward_ids.append(ids_all)
            entry=dict(request_id=key,placement=placement,forward=forward,loop=loop,
                phase='initial_prefill' if forward==0 else 'teacher_forced_cache_free_decode_prefix',
                paths=int(counts.sum()),active_experts=int((counts>0).sum()),
                expert_histogram=counts.tolist(),rank_loads=loads.tolist(),rank_max_mean=float(loads.max()/loads.mean()),
                rank_cv=float(loads.std()/loads.mean()),hottest_rank=int(loads.argmax()),
                expert_time_hottest_rank=int(np.argmax([x['expert_ms'] for x in times])),
                loop_local_interval_hottest_rank=int(np.argmax([sum(x.values()) for x in times])),
                loop_local_interval_max_ms=float(max(sum(x.values()) for x in times)),
                active_experts_per_rank=active,edge_density=float((edges>0).mean()),
                destination_fanout=float(edges.sum()/ids_all.shape[0]),
                remote_paths=sum(a['remote_paths'] for a in accounting),
                dispatch_bytes=sum(a['dispatch_bytes'] for a in accounting),
                combine_bytes=sum(a['combine_bytes'] for a in accounting),
                logical_dispatch_send_bytes_per_rank=[a['dispatch_bytes'] for a in accounting],
                logical_dispatch_recv_bytes_per_rank=(np.stack([a['remote'].sum(0) for a in accounting]).sum(0)*2048).tolist(),
                source_destination_row_counts=edges.tolist(),
                active_expert_ids_per_rank=[[e for e in range(64) if counts[e]>0 and mapping[e]==rank] for rank in range(4)],
                active_weight_bytes_proxy_per_rank=[n*9*1024**2 for n in active],
                rank_times=times,critical_rank=critical,
                S1_or_S0_max_rank_load=int(loads.max()),
                S2_candidate_max_rank_load=max(oracle_load),
                ideal_load_lower_bound=int(np.ceil(counts.sum()/4)),
                S2_candidate_placement=candidate)
            for field in ['dispatch_ms','expert_ms','combine_ms','restoration_ms']:
                entry[field]=times[critical][field]
            metrics.append(entry)
    adjacent=[jsd(histograms[h],histograms[h+1]) for h in range(15)]
    noise=[];overlap=[];matched_between=[]
    for h,ids in enumerate(first_forward_ids):
        order=np.random.default_rng(3407).permutation(len(ids));cut=len(ids)//2
        noise.append(jsd(np.bincount(ids[order[:cut]].ravel(),minlength=64),
                         np.bincount(ids[order[cut:]].ravel(),minlength=64)))
        if h<15:
            matched_between.append(jsd(
                np.bincount(ids[order[:cut]].ravel(),minlength=64),
                np.bincount(first_forward_ids[h+1][order[cut:]].ravel(),minlength=64)))
            overlap.append(float(np.mean([len(set(a)&set(b))/2 for a,b in zip(ids,first_forward_ids[h+1])])))
    base=[r for r in metrics if r['forward']==0]
    hot=[r['hottest_rank'] for r in base]
    row.update(adjacent_loop_JSD=float(np.mean(adjacent)),within_loop_split_JSD=float(np.mean(noise)),
        JSD_excess_over_split=float(np.mean(adjacent)-np.mean(noise)),
        matched_half_between_JSD=float(np.mean(matched_between)),
        matched_half_JSD_excess=float(np.mean(matched_between)-np.mean(noise[:15])),
        adjacent_same_token_overlap=float(np.mean(overlap)),
        hottest_rank_switch_rate=float(np.mean(np.diff(hot)!=0)),
        rank_max_mean=float(np.mean([r['rank_max_mean'] for r in metrics])),
        rank_cv=float(np.mean([r['rank_cv'] for r in metrics])),
        active_experts=float(np.mean([r['active_experts'] for r in metrics])),
        edge_density=float(np.mean([r['edge_density'] for r in metrics])),
        dispatch_bytes=sum(r['dispatch_bytes'] for r in metrics),
        combine_bytes=sum(r['combine_bytes'] for r in metrics),
        static_sum_max=sum(r['S1_or_S0_max_rank_load'] for r in metrics),
        candidate_sum_max=sum(r['S2_candidate_max_rank_load'] for r in metrics),
        ideal_sum_max=sum(r['ideal_load_lower_bound'] for r in metrics))
    return row,metrics


def main():
    frozen=json.loads((ROOT/'results/frozen_requests.json').read_text())
    static=json.loads((ROOT/'results/frozen_best_static.json').read_text())
    directory=ROOT/'results/raw/held_out_paired'
    assert json.loads((directory/'status.json').read_text())['success']
    rows=[];loops=[]
    for request in frozen['requests']:
        if request['split']!='held_out':continue
        for arm in ['S0','S1']:
            mapping=[e//16 for e in range(64)] if arm=='S0' else static['placement']
            row,details=episode(directory,request['request_id'],arm,mapping,request['request_id']+'_'+arm)
            row['prompt_length']=request['prompt_length'];rows.append(row);loops.extend(details)
    assert len(rows)==48
    comparisons=[]
    for request in sorted({r['request_id'] for r in rows}):
        a=next(r for r in rows if r['request_id']==request and r['placement']=='S0')
        b=next(r for r in rows if r['request_id']==request and r['placement']=='S1')
        comparisons.append(dict(request_id=request,
            physical_clean_reduction=1-b['clean_ms']/a['clean_ms'],
            measured_EP_reduction=1-b['sparse_EP_ms']/a['sparse_EP_ms'],
            physical_EP_plus_restore_reduction=1-b['sparse_EP_plus_restore_ms']/a['sparse_EP_plus_restore_ms'],
            structural_maxload_reduction=1-b['static_sum_max']/a['static_sum_max'],
            S1_to_S2_candidate_structural_headroom=1-b['candidate_sum_max']/b['static_sum_max'],
            S1_to_ideal_structural_upper=1-b['ideal_sum_max']/b['static_sum_max']))
    summary={}
    for arm in ['S0','S1']:
        observations=[r for r in rows if r['placement']==arm]
        fields=['clean_ms','event_ms','sparse_EP_ms','dispatch_ms','expert_ms','combine_ms','restoration_ms',
            'event_EP_share','observer_overhead','adjacent_loop_JSD','within_loop_split_JSD',
            'JSD_excess_over_split','matched_half_between_JSD','matched_half_JSD_excess',
            'adjacent_same_token_overlap','hottest_rank_switch_rate',
            'rank_max_mean','rank_cv','active_experts','edge_density']
        summary[arm]={key:bootstrap_mean([r[key] for r in observations]) for key in fields}
        observer=summary[arm]['observer_overhead']
        summary[arm]['event_timing_status']='UNTRUSTED' if observer['ci95'][1]>.10 or observer['mean']>.10 else 'VALIDATED_BOUNDED'
    contrasts={key:bootstrap_mean([r[key] for r in comparisons]) for key in comparisons[0] if key!='request_id'}
    strata={}
    for length in sorted({r['prompt_length'] for r in rows}):
        ids={r['request_id'] for r in rows if r['prompt_length']==length}
        paired=[r for r in comparisons if r['request_id'] in ids]
        strata[str(length)]={key:bootstrap_mean([r[key] for r in paired])
                            for key in paired[0] if key!='request_id'}
    write(ROOT/'results/physical_timing.json',dict(requests=rows,summary=summary,
        label='MEASURED_TRUE_EP4_NATIVE_CACHE_FREE_PREFIX_FORWARDS',
        forward_count_per_episode=3,headline_unit='ms / 3-forward episode, NOT cached token latency',
        critical_rule='one cumulative D+E+C+restoration critical rank per request/static arm across event repeats',
        comparison=contrasts,prompt_length_strata=strata,
        sample_unit='request episode; 2000 paired bootstrap; not loops/tokens/repeats'))
    write(ROOT/'results/placement_comparison.json',dict(
        physical_S0_S1=contrasts,
        S1_label=static['label'],S1_fit_requests=static['calibration_request_ids'],
        S2_label='FUTURE_TRUTH_GREEDY_LOAD_CANDIDATE; perfect-balance bound separately',
        S2_cost_domain='STRUCTURAL_ASSIGNMENT_PATHS_ONLY',
        S1_to_S2_EP_latency_headroom='UNVALIDATED; no calibrated latency oracle',
        non_executable_oracle=True,zero_migration_assumption=True,
        per_request=comparisons))
    write(ROOT/'results/validity.json',dict(true_EP4=True,physical_GPUS=GPUS,
        independent_request_episodes=24,static_arms=2,core_owned_per_rank=16,head_tail_owned_per_rank=2,
        native_H=16,all_logit_probe_comparisons=sum(r['total_logit_comparisons'] for r in rows),
        bitwise_logit_probe_comparisons=sum(r['exact_logit_comparisons'] for r in rows),
        router_ids_weights_and_residual_samples='EXACT_PASS',greedy_4_token_native_and_placement_parity='PASS',
        all_receive_paths_and_deduplicated_rows='PASS',
        representative_EP_checkpoint=False,native_KV_cache=False,
        S2_latency_oracle_validated=False,full_serving_runtime=False,
        loop_count_quality_sweep='NOT_RUN: changing trained H is not a supported primary control',
        accuracy_benchmark='NOT_RUN: tiny base model; logit/router/short greedy correctness only'))
    table(ROOT/'results/per_loop_rank_metrics.csv.gz',loops)
    table(ROOT/'results/per_loop_routing.csv.gz',[{k:r[k] for k in
          ['request_id','placement','forward','loop','expert_histogram','paths','active_experts']} for r in loops])
    print('REQUESTS',len(rows),'S0/S1 physical contrasts:',contrasts)


if __name__=='__main__':main()
