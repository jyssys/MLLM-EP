"""Report output-delivery metrics without substituting host service starts."""
import math
from collections import Counter
from .common import *


def percentile(values,q):
    if not values:return None
    values=sorted(values);p=(len(values)-1)*q;lo=int(p);hi=math.ceil(p)
    return values[lo]+(values[hi]-values[lo])*(p-lo)


def metrics(name):
    configure();folder=ROOT/'raw'/name;cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    receipt=folder/'dispatcher_result.json'
    if not receipt.exists() or not json.loads(receipt.read_text()).get('complete'):
        return dict(name=name,complete=False)
    rows=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    rows=[r for r in rows if not r.get('warmup')];assert len(rows)==cfg['count']
    slo=json.loads((ROOT/'configs/frozen_SLO.json').read_text())[cfg['model']+'_L'+str(cfg['length'])]
    ttft=[r['ttft_ms'] for r in rows];tpot=[r['tpot_ms'] for r in rows]
    itl=[]
    for row in rows:
        events=row['token_events']
        for previous,current in zip(events,events[1:]):
            itl.append(1000*(current['time']-previous['time'])/current['new_tokens'])
    start=min(r['arrival'] for r in rows);end=max(r['finish'] for r in rows);wall=end-start
    e2e=[1000*(r['finish']-r['arrival']) for r in rows]
    begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    offered=[json.loads(s) for s in (folder/'offered_requests.jsonl').read_text().splitlines()]
    sends=[r['actual_send_start'] for r in offered]
    events=[r for p in folder.glob('intervention_*.json') for r in json.loads(p.read_text()).get('records',[])
        if r['time']>=begin]
    commits=[r for r in events if r['event']=='SOURCE_COMMIT']
    decisions=[r for r in events if r['event']=='ADMISSION']
    from .qualification import measured_D_prompt_rows
    D_work=measured_D_prompt_rows(name)
    limit=65.08287642151119 if cfg['model']=='ouro' else 192.8212538175284
    ttft_limit=1112.9925646819174 if cfg['model']=='ouro' else 1963.9920117333531
    tpot_limit=53.70702898374411 if cfg['model']=='ouro' else 186.5715059141318
    result=dict(name=name,model=cfg['model'],method=cfg.get('intervention_method'),complete=True,
        count=len(rows),nominal_QPS=cfg['arrival_rate'],arrival_seed=cfg['arrival_seed'],
        realized_scheduled_QPS=(len(rows)-1)/(max(r['arrival'] for r in rows)-start) if len(rows)>1 and max(r['arrival'] for r in rows)>start else None,
        actual_submit_QPS=(len(sends)-1)/(max(sends)-min(sends)) if len(sends)>1 and max(sends)>min(sends) else None,
        completed_QPS=len(rows)/wall,output_tokens_s=sum(r['output_tokens'] for r in rows)/wall,
        input_tokens_s=sum(r['input_tokens'] for r in rows)/wall,
        SLO_goodput=sum(r['ttft_ms']<=ttft_limit and r['tpot_ms']<=tpot_limit for r in rows)/wall,
        TTFT_attainment=sum(x<=ttft_limit for x in ttft)/cfg['count'],
        TPOT_attainment=sum(x<=tpot_limit for x in tpot)/cfg['count'],
        ITL_limit_misses=sum(x>limit for x in itl),ITL_intervals=len(itl),maximum_ITL_ms=max(itl,default=None),
        coalesced_nonfirst_delivery_events=sum(not e['first'] and e['new_tokens']>1 for r in rows for e in r['token_events']),
        maximum_TTFT_ms=max(ttft),maximum_E2E_ms=max(e2e),
        ITL_contract='Frozen event-average delivery intervals. Exact individual token-generation times are not measured when delivery is coalesced.',
        offloaded_requests=len(D_work),actual_D_prompt_recurrent_rows=sum(D_work.values()),
        accepted_loop_counts=dict(Counter(r['remaining'] for r in commits if not r.get('token_split'))),
        admissions=len(decisions),rejected_admissions=sum(not r['accepted'] for r in decisions),
        Service_TTFT='NOT_MEASURED' if not cfg.get('observe_service_timing') else 'INSTRUMENTED_SEPARATE_ANALYSIS',
        latency_scope='Marker-free output-delivery metrics unless the configuration enables device events or state hashing.',
        throughput_scope='Finite-cohort drain rates. Not sustainable serving capacity.')
    for label,values in (('TTFT',ttft),('TPOT',tpot),('ITL',itl),('E2E',e2e)):
        for q in (.5,.95,.99):result[f'{label}_p{int(q*100)}_ms']=percentile(values,q)
    result['Decode_SLO_screen']=result['TPOT_p95_ms']<=.9*tpot_limit and result['ITL_p99_ms']<=limit
    result['frozen_SLO']=slo
    write(folder/'intervention_metrics.json',result)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    print(json.dumps(metrics(a.name),indent=2))
