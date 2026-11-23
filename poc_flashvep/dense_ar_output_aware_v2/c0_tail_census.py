"""Audit all heldout C0 client SLO intervals without changing timestamps."""
import collections
import json
import time
from .common import ROOT, write, digest
from .c0_analysis import csv_save


def main():
    plan=json.loads((ROOT/'jobs/c0_output_matrix_v2.json').read_text())
    jobs={b[arm]:(b,arm) for b in plan['blocks'] if b['split']=='heldout' for arm in ('Native','Layer','Token')}
    runs=[];violations=[];groups=collections.defaultdict(collections.Counter)
    for name,(block,arm) in jobs.items():
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        if not receipt.exists() or not json.loads(receipt.read_text())['complete']:continue
        requests=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
        intervals=0;count=collections.Counter();maximum=None
        for request in requests:
            events=request['token_events'];intervals+=request['output_tokens']-1
            for previous,current in zip(events,events[1:]):
                client=1000*(current['time']-previous['time'])
                if client<=34:continue
                worker=1000*(current['worker_time']-previous['worker_time'])
                delivery=1000*((current['time']-current['worker_time'])-(previous['time']-previous['worker_time']))
                assert abs(client-worker-delivery)<1e-4
                key='worker_gap_over34' if worker>34 else 'worker_gap_at_most34'
                count[key]+=1
                row=dict(name=name,arm=arm,input_bucket=block['input_bucket'],output=block['output'],start=block['start'],
                    sample_id=request['sample_id'],prompt_id=request['prompt_id'],ITL_ms=client,
                    worker_gap_ms=worker,delivery_delay_delta_ms=delivery,worker_gap_over34=worker>34,
                    evidence='TRACE_ANALYSIS')
                violations.append(row)
                if maximum is None or client>maximum['ITL_ms']:maximum=row
        metric=json.loads((ROOT/'analysis'/(name+'_metrics.json')).read_text())
        assert intervals==metric['ITL_interval_count']
        runs.append(dict(name=name,arm=arm,input_bucket=block['input_bucket'],output=block['output'],start=block['start'],
            requests=len(requests),intervals=intervals,violations=sum(count.values()),**dict(count),
            max_ITL_ms=metric['max_ITL_ms'],ITL_p99_ms=metric['ITL_p99_ms'],TPOT_p95_ms=metric['TPOT_p95_ms'],
            every_ITL_request_attainment=metric['every_ITL_SLO_attainment'],maximum_violation=maximum,
            requests_SHA256=digest(folder/'requests.jsonl')))
        group=groups[arm,block['output']];group.update(count)
        group.update(independent_runs=1,requests=len(requests),intervals=intervals,
            requests_failing_all_ITL=sum(any(1000*(b['time']-a['time'])>34 for a,b in zip(r['token_events'],r['token_events'][1:])) for r in requests))
    summary=[dict(arm=arm,output=output,**dict(value)) for (arm,output),value in sorted(groups.items())]
    write(ROOT/'analysis/c0_heldout_client_tail_census_v2.json',dict(time=time.time(),runs=runs,summary=summary,
        scope='All completed unique heldout C0 runs. A Native run reused across cuts is counted once. '
            'Client ITL equals worker-emission gap plus delivery-delay change. No timestamp is corrected. '
            'A client violation with worker gap<=34ms locates the violation in delivery-delay growth. '
            'This identity does not identify a blocked syscall,GC,CPU scheduling or GPU operator cause. '
            'Counts across runs are descriptive,not a server-start confidence interval. C0 is not C3 continuous serving.'))
    csv_save(ROOT/'figures/c0_heldout_client_tail_runs_v2.csv',runs)
    csv_save(ROOT/'figures/c0_heldout_client_tail_violations_v2.csv',violations)
    csv_save(ROOT/'figures/c0_heldout_client_tail_summary_v2.csv',summary)
    print('C0_ALL_HELDOUT_CLIENT_TAIL_CENSUS',len(runs),'runs',len(violations),'violations',flush=True)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
