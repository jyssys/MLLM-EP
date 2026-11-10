"""Account for every request and compare independent paired serving starts."""
import csv
import math
import random
from collections import Counter,defaultdict
from .common import *
from .analyze import metrics,qualify,percentile
from .audit_old import events


def csv_write(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def inspect(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    item=metrics(name)
    if not item.get('complete'):
        receipt=folder/'dispatcher_result.json'
        item.update(model=cfg['model'],method=cfg['intervention_method'],count=cfg['count'],
            arrival_seed=cfg['arrival_seed'],
            status=json.loads(receipt.read_text()).get('status') if receipt.exists() else 'NOT_RUN')
        return item,[],[]
    begin=json.loads((folder/'measurement_begin.json').read_text())['time'];ev=events(folder,begin)
    reqs=[r for r in map(json.loads,(folder/'requests.jsonl').read_text().splitlines()) if not r.get('warmup')]
    request_metrics=json.loads((folder/'stats.json').read_text())['metrics']
    by_rid={r['request_id']:r for r in request_metrics};by_tid={r['transfer_id']:r for r in request_metrics}
    arms=defaultdict(list)
    for r in ev:
        if r['event']=='SOURCE_ARM_DECISION':arms[r['seed']].append(r)
    commits={r['seed']:r for r in ev if r['event']=='SOURCE_COMMIT'}
    rows=[];completed=[]
    for req in reqs:
        m=by_rid[f"r0_q{req['sample_id']}"];tid=m['transfer_id'];ad=[r for r in ev if r['event']=='ADMISSION' and r['tid']==tid]
        cs=commits.get(req['seed']);evaluations=arms[req['seed']]
        token_selection=[r for r in ev if r['event']=='TOKEN_INITIAL_BOUNDARY_SELECTION' and r['tid']==tid]
        eligible=any(r['armed'] for r in evaluations)
        armed=any(r['armed'] for r in token_selection) if token_selection else eligible
        rows.append(dict(name=name,model=cfg['model'],arm=cfg.get('primary_safety_arm'),arrival_seed=cfg['arrival_seed'],
            sample_id=req['sample_id'],sampling_seed=req['seed'],tid=tid,arm_evaluations=len(evaluations),
            source_monitor_eligible=eligible,armed=armed,
            boundary_offers=sum(r['event']=='GPU_COMPLETED_BOUNDARY_OFFER' and r['tid']==tid for r in ev),
            admission_evaluations=len(ad),prepared=sum(r['accepted'] for r in ad),committed=bool(cs),
            offloaded_full_prompt_loops=cs['remaining'] if cs and not cs.get('token_split') else 0,
            offloaded_suffix_tokens=cs['remaining'] if cs and cs.get('token_split') else 0,
            final_request_TTFT_ms=req['ttft_ms'],final_request_E2E_ms=req['e2e_ms'],
            rejection_reasons='|'.join(str(r.get('rejection_reason')) for r in ad if not r['accepted'])))
        if not cs:continue
        quanta=sorted((r for r in ev if r['event']=='D_QUANTUM_HOST_COMPLETION' and
            (r.get('choice') or {}).get('tid')==tid),key=lambda r:r['submitted_host_time'])
        plan=next((r for r in reversed(ev) if r['event']=='COMPLETION_PLAN' and r['tid']==tid and
            r['epoch']==cs['epoch'] and r['feasible']),{})
        origin=m['created_monotonic'];ts=m['timings'];received=origin+ts['d_received'] if 'd_received' in ts else None
        completed.append(dict(name=name,model=cfg['model'],arrival_seed=cfg['arrival_seed'],tid=tid,seed=cs['seed'],
            token_split=cs.get('token_split'),remaining=cs['remaining'],
            predicted_completion_ms=plan.get('predicted_ms'),reserved_deadline_ms=plan.get('deadline_ms'),
            committed_to_first_output_host_ms=1000*(req['first_token']-cs['time']),
            first_output_deadline_overrun_observed=(1000*(req['first_token']-cs['time'])>plan['deadline_ms'])
                if plan.get('deadline_ms') is not None else None,
            first_output_prediction_error_ms=(1000*(req['first_token']-cs['time'])-plan['predicted_ms'])
                if plan.get('predicted_ms') is not None else None,
            import_to_first_output_host_ms=1000*(req['first_token']-received) if received else None,
            commit_to_first_quantum_host_ms=1000*(quanta[0]['submitted_host_time']-cs['time']) if quanta else None,
            largest_inter_quantum_wait_ms=max((1000*(b['submitted_host_time']-a['time']) for a,b in zip(quanta,quanta[1:])),default=None),
            submitted_quanta=len(quanta),host_quantum_enclosure_sum_ms=sum(r['elapsed_host_ms'] for r in quanta),
            observed_completion_reservation_missed=any(r['event']=='COMPLETION_RESERVATION_MISSED' and r['tid']==tid for r in ev),
            scope='Host observations include submission, polling, communication, and Decode scheduling. They are not GPU kernel occupancy.'))
    item['source_armed_requests']=sum(r['armed'] for r in rows)
    item['source_arm_evaluations']=sum(r['arm_evaluations'] for r in rows)
    item['boundary_offers']=sum(r['boundary_offers'] for r in rows)
    item['completed_full_prompt_loops']=sum(r['offloaded_full_prompt_loops'] for r in rows)
    item['rejection_counts']=dict(Counter(r.get('rejection_reason') for r in ev if r['event']=='ADMISSION' and not r['accepted']))
    item['completion_reservation_misses']=sum(r['event']=='COMPLETION_RESERVATION_MISSED' for r in ev)
    item['first_output_deadline_overruns_observed']=sum(r['first_output_deadline_overrun_observed'] is True for r in completed)
    item['device_compute_time_saved_ms']='NOT_MEASURED'
    item['graph_cache']={}
    for p in folder.glob('deep_graph_cache_*.json'):
        g=json.loads(p.read_text());item['graph_cache'][g['role']]=g['counts']
    # A cold measured graph can disturb tails. Report this confound explicitly.
    item['measured_graph_captures']=sum(g['measured']['captures'] for g in item['graph_cache'].values())
    frozen=OLD/'raw'/f"confirm_{cfg['model']}_native_seed2026110911"/'requests.jsonl'
    if frozen.exists():
        native={r['seed']:r for r in map(json.loads,frozen.read_text().splitlines()) if not r.get('warmup')}
        different=[r['seed'] for r in reqs if r['seed'] not in native or any(
            r[k]!=native[r['seed']][k] for k in ('input_ids','output_ids','exit_depths'))]
        item['frozen_Native_output']=dict(reference=str(frozen),count=len(reqs),divergent_seeds=different,passed=not different)
    item['N128_full_state_hash_scope']='NOT_MEASURED_IN_MARKER_FREE_CONFIRMATION. N8/N32 asynchronous full-state diagnostics qualify the path.'
    reference=cfg.get('comparison_reference')
    if reference and (ROOT/'raw'/reference/'requests.jsonl').exists():
        item['qualification']=qualify(name,reference)
    else:item['qualification']=qualify(name)
    return item,rows,completed


def paired(items,arms=('old','loop','token'),references=('native','token')):
    grouped=defaultdict(dict)
    for r in items:
        if not r.get('complete'):continue
        arm=r['name'].split('_')[2];grouped[r['model'],r['arrival_seed']][arm]=r
    results=[]
    for model in ('ouro','huginn'):
        for reference in references:
            for arm in arms:
                if arm==reference:continue
                for key in ('TTFT_p95_ms','E2E_p95_ms','ITL_p99_ms','maximum_ITL_ms','TPOT_p95_ms','completed_QPS','SLO_goodput'):
                    values=[];sources=[];differences=[];zero_references=[]
                    for (m,seed),runs in sorted(grouped.items()):
                        if m!=model or arm not in runs or reference not in runs:continue
                        a,b=runs[arm].get(key),runs[reference].get(key)
                        if a is None or b is None:continue
                        assert a>=0 and b>=0
                        values.append(a/b if b else None);sources.append(seed);differences.append(a-b)
                        if not b:zero_references.append(seed)
                    if not values:continue
                    def geometric(v):
                        if any(x is None for x in v):return None
                        return 0. if any(x==0 for x in v) else math.exp(sum(math.log(x) for x in v)/len(v))
                    point=geometric(values);ci=None;delta_ci=None
                    if len(values)>=3 and not zero_references:
                        rng=random.Random(2026111017)
                        samples=[geometric([rng.choice(values) for _ in values]) for _ in range(10000)]
                        ci=[percentile(samples,.025),percentile(samples,.975)]
                    if len(differences)>=3:
                        rng=random.Random(2026111017)
                        samples=[sum(rng.choice(differences) for _ in differences)/len(differences) for _ in range(10000)]
                        delta_ci=[percentile(samples,.025),percentile(samples,.975)]
                    results.append(dict(model=model,arm=arm,reference=reference,metric=key,paired_starts=len(values),
                        seeds=sources,start_ratios=values,geometric_ratio=point,bootstrap_95_CI=ci,
                        zero_reference_seeds=zero_references,paired_mean_difference=sum(differences)/len(differences),
                        mean_difference_bootstrap_95_CI=delta_ci,
                        ratio_status='UNDEFINED_ZERO_REFERENCE' if zero_references else 'DEFINED_WITH_ZEROS_RETAINED',
                        interpretation='Paired start-level bootstrap. Three starts provide limited confidence coverage. No token-level independence assumption.'))
    return results


def main():
    configure();items=[];requests=[];completions=[]
    for name in json.loads((ROOT/'primary_plan.json').read_text())['jobs']:
        folder=ROOT/'raw'/name
        if not (folder/'dispatcher_result.json').exists():
            item=dict(name=name,complete=False,status='NOT_RUN')
        else:
            item,rows,finished=inspect(name);requests.extend(rows);completions.extend(finished)
        items.append(item)
    write(ROOT/'serving_results.json',items);csv_write(ROOT/'serving_results.csv',items)
    write(ROOT/'request_gate_inventory.json',requests);csv_write(ROOT/'request_gate_inventory.csv',requests)
    write(ROOT/'completion_model_audit.json',completions);csv_write(ROOT/'completion_model_audit.csv',completions)
    ratios=paired(items);write(ROOT/'paired_start_ratios.json',ratios);csv_write(ROOT/'paired_start_ratios.csv',ratios)
    print(json.dumps(dict(starts=len(items),complete=sum(r.get('complete',False) for r in items),
        accounted_requests=len(requests),completed_migrations=len(completions))))


if __name__=='__main__':main()
