"""Check simulated burst operator counts against physical native CDB executions."""
from collections import Counter
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_decode_first_depth_aware_dp_v3.simulator import CostModel,simulate_dp
from poc_decode_first_depth_aware_dp_v3.scripts.analyze_live import simulated_inputs

def main():
    source=ROOT/'outputs/live_trace_v1'
    assert read(source/'status.json')['success']
    model=CostModel({s:dict(coefficients=[1.]+[0.]*(n-1),minimum_ms=1.)
        for s,n in [('prefill',4),('recurrent',5),('coda',2)]})
    rows=[]
    for meta in sorted(source.glob('trace_burst*.json')):
        data=read(meta)
        case=data['case']
        ranks=[read(source/f'rank_{i}'/meta.name) for i in range(4)]
        inputs=simulated_inputs(case,ranks,data['assignments'])
        pred=simulate_dp(inputs,model)
        for i,(actual,sim) in enumerate(zip(ranks,pred['ranks'])):
            counts=Counter(s['stage'] for s in sim['stages'])
            matched=all(counts[s]==actual['stats'][s+'_batches'] for s in ['prefill','recurrent','coda'])
            rows.append(dict(case=case['name'],rank=i,counts=dict(counts),
                actual={s:actual['stats'][s+'_batches'] for s in ['prefill','recurrent','coda']},matched=matched))
    write(ROOT/'artifacts/transition_validation.json',dict(rows=rows,
        all_match=all(r['matched'] for r in rows),
        interpretation='Only transition counts, not timing validation. Unit-cost simulation is not a speedup estimate.'))
    print('BURST_TRANSITION_COUNTS',sum(r['matched'] for r in rows),'/',len(rows))

if __name__=='__main__':main()
