"""Analyze finite fixed-cut cohorts. Do not infer sustainable serving capacity."""
import csv
import json
import statistics
from collections import defaultdict
from .common import ROOT,write
from .gate import read


def quantile(values,q):
    values=sorted(values)
    if not values:return 'NOT_MEASURED'
    index=(len(values)-1)*q;left=int(index);fraction=index-left
    return values[left]*(1-fraction)+values[min(left+1,len(values)-1)]*fraction


def csv_write(name,rows):
    if not rows:return
    with (ROOT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def main():
    summaries=[];parity=[];phases=[];work=[];request_rows=[]
    for folder in sorted((ROOT/'raw').glob('map_*')):
        status=folder/'status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':continue
        cfg=json.loads((folder/'arc_contract.json').read_text())
        rows=[r for r in read(folder/'requests.jsonl') if not r['warmup']]
        by_cohort=defaultdict(list)
        for row in rows:by_cohort[row['cohort_id']].append(row)
        references={(r['regime'],r['seed']):r for r in rows if r['cut']==r['recurrent_depth']}
        for key,cohort in by_cohort.items():
            targets=[r for r in cohort if r['target']]
            first=targets[0];depth=first['recurrent_depth'];cut=first['cut']
            window=json.loads((folder/f'window_{key}.json').read_text());seconds=window['end']-window['start']
            intervals=[1000*(e['time']-e['previous_time']) for r in targets for e in r['token_events'] if not e['first'] and e['new_tokens']==1]
            row=dict(name=folder.name,model=cfg['model'],start=cfg['repeat'],gpus=','.join(map(str,cfg['gpus'])),
                     cohort=key,regime=first['regime'],cut=cut,prompt_tokens=first['input_tokens'],
                     target_requests=len(targets),all_requests=len(cohort),window_s=seconds,
                     finite_requests_per_s=len(cohort)/seconds,finite_output_tokens_per_s=sum(r['output_tokens'] for r in cohort)/seconds,
                     evidence='MEASURED_FINITE_COHORT',sustainable_capacity='NOT_MEASURED')
            for field in ('ttft_ms','tpot_ms','e2e_ms'):
                values=[r[field] for r in targets]
                row[field+'_mean']=statistics.mean(values)
                for q in (.5,.95,.99):row[field+f'_p{int(q*100)}']=quantile(values,q)
            for q in (.5,.95,.99):row[f'observed_ITL_ms_p{int(q*100)}']=quantile(intervals,q)
            row['maximum_observed_ITL_ms']=max(intervals) if intervals else 'NOT_MEASURED'
            row['observed_single_token_intervals']=len(intervals)
            row['multi_token_notifications']=sum(e['new_tokens']>1 for r in targets for e in r['token_events'])
            summaries.append(row)
            statistics_path=folder/f'cohort_{key}_stats.json'
            counters=json.loads(statistics_path.read_text())
            before,after=counters['before'],counters['after']
            total_rows=sum(after[k]['issued_loop_rows']-before[k]['issued_loop_rows'] for k in before)
            required=sum((r['input_tokens']+r['output_tokens']-1)*depth for r in cohort)
            work.append(dict(name=folder.name,cohort=key,issued_loop_rows=total_rows,required_loop_rows=required,passed=total_rows==required))
            metrics={m['request_id']:m for m in counters['metrics']}
            for r in targets:
                ref=references.get((r['regime'],r['seed']))
                exact=bool(ref and all(r[k]==ref[k] for k in ('input_ids','output_ids','exit_depths','finish_reason')))
                parity.append(dict(name=folder.name,cohort=key,model=cfg['model'],seed=r['seed'],cut=r['cut'],passed=exact))
                request_rows.append({k:r[k] for k in ('name','cohort_id','model','regime','cut','repeat','sample_id','input_tokens','output_tokens','ttft_ms','tpot_ms','e2e_ms','finish_reason')})
                m=metrics.get(f"r{r['run']}_q{r['sample_id']}")
                if not m:continue
                t=m['timings'];created=m['created_monotonic']
                ttft=r['arrival']+r['ttft_ms']/1000
                if cut:
                    keys=['harness_delivery','coordinator_P_queue','P_interval_observed','exposed_handoff_observed','D_wait_and_continuation_to_first_token','generation_after_first_token']
                    points=[r['arrival'],r['admission'],created+t['prefill_started'],created+t['prefill_compute_completed'],created+t['d_received'],ttft,r['finish']]
                else:
                    keys=['harness_delivery','coordinator_D_reservation','D_prefill_and_queue_to_first_token','generation_after_first_token']
                    points=[r['arrival'],r['admission'],created+t['d_reserved'],ttft,r['finish']]
                if any(b<a for a,b in zip(points,points[1:])):continue
                for phase,left,right in zip(keys,points,points[1:]):
                    phases.append(dict(name=folder.name,cohort=key,model=cfg['model'],cut=cut,sample_id=r['sample_id'],
                                       component=phase,ms=1000*(right-left),share_pct=100*(right-left)/(r['finish']-r['arrival']),
                                       denominator='One request E2E. Exclusive host-observed intervals. Not GPU kernel shares.',
                                       evidence='MEASURED_HOST_OBSERVATION'))
    csv_write('static_cut_metrics_by_start.csv',summaries)
    csv_write('request_latency_raw.csv',request_rows)
    csv_write('map_output_parity.csv',parity)
    csv_write('map_full_depth_work.csv',work)
    csv_write('e2e_component_shares.csv',phases)
    write(ROOT/'state/map_validation.json',dict(completed_starts=len({r['name'] for r in summaries}),
          cells=len(summaries),paired_requests=len(parity),output_exact=sum(r['passed'] for r in parity),
          work_checks=len(work),work_exact=sum(r['passed'] for r in work),
          passed=bool(parity) and all(r['passed'] for r in parity) and all(r['passed'] for r in work)))
    print('MAP_CELLS',len(summaries),'PARITY',sum(r['passed'] for r in parity),'/',len(parity))


if __name__=='__main__':main()
