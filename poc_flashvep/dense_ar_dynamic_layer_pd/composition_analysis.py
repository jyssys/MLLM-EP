"""Reconstruct Host-visible Decode concurrency without inferring GPU idle."""
import collections
from .common import *
from .analyze import records


def union(intervals):
    result=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1][1]=max(result[-1][1],b)
        else:result.append([a,b])
    return result


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    req={f'q{r["sample_id"]}':r for r in records(folder/'requests.jsonl')}
    window=json.loads((folder/'window.json').read_text())
    assigned={f'q{r["sample_id"]}':r['role'] for r in records(folder/'actual_submissions.jsonl') if 'role' in r}
    roles=('replica0','replica1') if cfg['arm']=='DP2' else ('decode',)
    results=[]
    for role in roles:
        users=[r for rid,r in req.items() if cfg['arm']!='DP2' or assigned.get(rid)==role]
        assert users
        events=sorted([(r['first_token'],1) for r in users]+[(r['finish'],-1) for r in users])
        count=0;last=window['start'];durations=collections.defaultdict(float)
        for timestamp,change in events:
            durations[count]+=timestamp-last;count+=change;last=timestamp;assert count>=0
        durations[count]+=window['end']-last;assert count==0
        active=union([(r['first_token'],r['finish']) for r in users])
        observed=json.loads(next(folder.glob(f'native_{role}_*.json')).read_text())
        samples=[s for s in observed['queue_samples'] if window['start']<=s['time']<=window['end']]
        scheduled=[r['time'] for r in observed['prompt_spans']
            if r['request_id'].split('-')[0] in req]
        steps=union([(s['start'],s['end']) for s in observed['host_step_spans']
            if any(s['start']<=t<=s['end'] for t in scheduled)])
        overlap=sum(max(0,min(b,d)-max(a,c)) for a,b in steps for c,d in active)
        positive=sum(v for k,v in durations.items() if k>0)
        results.append(dict(role=role,requests=len(users),
            Host_user_Decode_concurrency_seconds=dict(durations),
            Host_user_Decode_concurrency_max=max(k for k,v in durations.items() if v>0),
            Host_user_Decode_concurrency_mean_when_positive=sum(k*v for k,v in durations.items())/positive,
            native_scheduler_running_sample_histogram=dict(collections.Counter(s['running'] for s in samples)),
            sampled_KV_usage_max=max((s['KV_usage'] for s in samples),default=None),
            measured_prompt_Host_step_count=len(steps),
            prompt_Host_step_overlap_with_same_role_user_Decode_s=overlap,
            scope='User Decode concurrency is first-to-last delivered-token occupancy. Scheduler running samples can include Prefill or KV wait. Neither quantity is an actual GEMM batch or a physical GPU busy/idle measurement. Host prompt-step overlap is not exclusive kernel overlap.'))
    return dict(name=name,roles=results,evidence='TRACE_ANALYSIS_HOST_COMPOSITION',
        causal_batch_or_clock_contribution='NOT_IDENTIFIED')


def main():
    names=[r['name'] for r in json.loads((ROOT/'analysis/heldout_results.json').read_text())['results']]
    extra=ROOT/'analysis/exploratory_fraction50_results.json'
    if extra.exists():names+=[r['name'] for r in json.loads(extra.read_text())['results']]
    results=[analyze(name) for name in names]
    write(ROOT/'analysis/Host_composition_and_overlap.json',dict(results=results,
        scope='Same-role Host observations distinguish cohort composition from direct kernel interference. No causal contribution percentage or GPU slack follows from this reconstruction.'))
    print('HOST_COMPOSITION',len(results))


if __name__=='__main__':main()
