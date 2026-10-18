"""Use calibration outputs only for causal output-length and cost priors."""
import statistics
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    assert read(ROOT/'outputs/discovery128_graph_b1_v1/status.json')['success']
    rows=read(ROOT/'artifacts/discovery_frozen.json')['rows']
    observations={}
    for path in (ROOT/'outputs/discovery128_graph_b1_v1').glob('*.json'):
        v=read(path)
        if isinstance(v,dict) and 'mean_depth' in v:observations[v['id']]=v
    priors={}
    for domain in ['gsm8k','math500','alpaca','sharegpt']:
        values=[observations[r['id']] for r in rows if r['dataset']==domain and r['split']=='calibration']
        priors[domain]=dict(output=statistics.median(v['output_length'] for v in values),
            ms_per_decode_token=statistics.median(v['decode_ms']/max(1,v['output_length']-1) for v in values))
    features={r['id']:dict(prompt_tokens=observations[r['id']]['prompt_tokens'],
        expected_output_tokens=priors[r['dataset']]['output'],
        expected_work_ms=priors[r['dataset']]['output']*priors[r['dataset']]['ms_per_decode_token']+
            observations[r['id']]['prompt_tokens']*.02) for r in rows}
    heldout=[r['id'] for r in rows if r['split']=='heldout']
    calibration=[r['id'] for r in rows if r['split']=='calibration']
    common=dict(local_batch_cap=16,causal_features=features,calibration_priors=priors,
        rank_rates=[1.,1.,1.,1.],
        information_boundary='No held-out output length, exit depth, token ID, or measured service time enters causal placement features.',
        expected_cost='Calibration median decode cost per domain and known prompt length. Equal nominal H100 rank service rates.',
        physically_pinned=True,cap=256)
    trace=[]
    for c in [1,2,4,8,16,32,64]:
        trace.append(dict(name=f'trace_burst_C{c}_rr',concurrency=c,arrival='burst',policy='round_robin',
            ids=calibration[:c] if c<=32 else heldout[:c],cost_fit=c<=32))
    trace.append(dict(name='trace_cost_local_B16',concurrency=16,arrival='burst',policy='single_rank',
        ids=calibration[:16],cost_fit=True))
    for arrival in ['closed_loop','poisson']:
        trace.append(dict(name=f'trace_{arrival}_C16_queue',concurrency=16,arrival=arrival,policy='queue',ids=heldout[:32],rate_rps=1.))
    clean=[]
    for c in [1,2,4,8,16,32,64]:
        for policy in ['round_robin','queue','length','strong_cost']:
            clean.append(dict(name=f'clean_burst_C{c}_{policy}',concurrency=c,arrival='burst',policy=policy,ids=heldout[:c]))
    for arrival in ['closed_loop','poisson']:
        for policy in ['round_robin','queue','length','strong_cost']:
            clean.append(dict(name=f'clean_{arrival}_C16_{policy}',concurrency=16,arrival=arrival,policy=policy,ids=heldout[:32],rate_rps=1.))
    for filename,cases in [('live_trace_plan.json',trace),('live_clean_plan.json',clean)]:
        path=ROOT/'artifacts'/filename
        assert not path.exists()
        write(path,dict(**common,cases=cases,
            limit='Poisson C16 denotes the nominal setting, not a guaranteed active-request count. Actual concurrency is logged.',
            heldout_discovery_traces_not_used_for_causal_features=True))

if __name__=='__main__':main()
