"""Bound retired local-prefill progress during existing native decode activity."""
from .common import *
from .analyze import requests,pct,table
from .profile_analysis import union


def contains(intervals,left,right):
    return any(a<=left<=right<=b for a,b in intervals)


def analyze(name):
    folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    rows=requests(folder)
    if not rows:return None
    native=[r for r in rows if r['sample_id']%cfg.get('dlocal_every',4)]
    periods=union([(r['first_token'],r['finish']) for r in native])
    tids={};selected=set()
    for path in folder.glob('dlocal_events_*.json'):
        for e in json.loads(path.read_text())['events']:
            if e['event']=='PLACEMENT' and e.get('local') and e['request_id'].startswith('r0_q'):
                selected.add(e['request_id'])
            if e['event']=='LOCAL_CREDIT_RESERVED' and e['request_id'].startswith('r0_q'):
                tids[e['tid']]=(e['request_id'],e['time'])
    frames=[]
    window=json.loads((folder/'window.json').read_text())
    for path in folder.glob('memory_progress_*.jsonl'):
        for line in path.read_text().splitlines():
            v=json.loads(line)
            if v['role']=='decode' and window['start']<=v['time']<=window['end']:frames.append(v)
    previous={};proof=[]
    for v in sorted(frames,key=lambda v:v['time']):
        unretired=set(v['all_submitted_unretired_ids'])
        for r in v['requests']:
            tid=r['request_id']
            if tid not in tids or tid in unretired:continue
            rid,admitted=tids[tid]
            before_time,before_count=previous.get(tid,(admitted,0))
            if r['prefilled']>before_count and contains(periods,before_time,v['time']):
                proof.append(dict(request_id=rid,begin=before_time,end=v['time'],
                    retired_prompt_tokens_lower_bound=r['prefilled']-before_count,
                    recurrent_token_rows_lower_bound=(r['prefilled']-before_count)*cfg['depth']))
            previous[tid]=(v['time'],r['prefilled'])
    proof_periods=union([(v['begin'],v['end']) for v in proof]);itls=[];tpots=[]
    for r in native:
        if any(r['first_token']<=b and r['finish']>=a for a,b in proof_periods):tpots.append(r['tpot_ms'])
        for e in r['token_events']:
            if not e['first'] and e['new_tokens']==1 and any(e['previous_time']<=b and e['time']>=a for a,b in proof_periods):
                itls.append(1000*(e['time']-e['previous_time']))
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())[f'{cfg["model"]}_L{cfg["length"]}']
    value=dict(name=name,model=cfg['model'],completed_requests=len(rows),selected_local_requests=len(selected),
        proof_windows=len(proof),retired_prompt_tokens_lower_bound=sum(v['retired_prompt_tokens_lower_bound'] for v in proof),
        retired_recurrent_rows_lower_bound=sum(v['recurrent_token_rows_lower_bound'] for v in proof),
        native_overlap_TPOT_p95_ms=pct(tpots,95),native_overlap_ITL_p99_ms=pct(itls,99),native_overlap_max_ITL_ms=max(itls,default=None),
        native_overlap_ITL_samples=len(itls),native_overlap_ITL_deadline_misses=sum(t>slos['ITL_ms'] for t in itls),
        sampled_overlap_screen=bool(itls and tpots and pct(tpots,95)<=.9*slos['TPOT_ms'] and pct(itls,99)<=slos['ITL_ms']),
        evidence='INFERRED_LOWER_BOUND_FROM_MEASURED_HOST_RETIREMENT_SNAPSHOTS',
        scope='Counts only positive final-depth prompt-frontier changes between settled snapshots with no unretired ticket for that local request. Each complete snapshot interval lies inside existing native decode activity. This is not a kernel timeline, exclusive GPU time, or guaranteed safe slack. Zero is not proof of absent overlap.')
    write(folder/'sampled_retired_overlap.json',dict(value,windows=proof))
    return value


def main():
    results=[]
    for path in sorted((ROOT/'configs').glob('serve_*.json')):
        cfg=json.loads(path.read_text());folder=ROOT/'raw'/cfg['name']
        if cfg.get('dlocal_method')=='native' or not (folder/'after_measured.json').exists():continue
        value=analyze(cfg['name'])
        if value:results.append(value)
    table(ROOT/'sampled_retired_overlap.csv',results)
    print(json.dumps(results))


if __name__=='__main__':main()
