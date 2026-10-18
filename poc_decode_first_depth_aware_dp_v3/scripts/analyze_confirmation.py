"""Paired live hindsight-placement comparison. Preserve output changes as a separate result."""
import numpy as np
from poc_decode_first_depth_aware_dp_v3.contract import *
from poc_decode_first_depth_aware_dp_v3.scripts.analyze_live import measured_case,simulated_inputs
from poc_decode_first_depth_aware_dp_v3.simulator import CostModel,simulate_dp

def main():
    model=CostModel(read(ROOT/'artifacts/cdb_cost_model.json')['parameters'])
    results=[]
    detailed=[]
    for suffix in ['', '_c64']:
        folder=ROOT/('outputs/hindsight_confirmation'+suffix+'_v1')
        assert read(folder/'status.json')['success']
        plan=read(ROOT/('artifacts/hindsight_confirmation'+suffix+'_frozen.json'))
        for c in sorted({r['concurrency'] for r in plan['cases']}):
            blocks={}
            route_validation=[]
            for case in plan['cases']:
                if case['concurrency']!=c:continue
                metric,ranks=measured_case(folder,case)
                outputs={r['id']:r for rank in ranks for r in rank['outputs']}
                blocks[case['repeat'],case['arm']]=(metric,outputs)
                detailed.append(dict(**metric,arm=case['arm'],repeat=case['repeat']))
                controller=read(folder/(case['name']+'.json'))
                own=simulate_dp(simulated_inputs(case,ranks,controller['assignments']),model)
                route_validation.append(dict(arm=case['arm'],repeat=case['repeat'],
                    predicted_from_actual_online_depth_P95_ms=own['E2E_P95_ms'],
                    measured_P95_ms=metric['E2E_P95_ms'],
                    relative_error_percent=100*abs(own['E2E_P95_ms']-metric['E2E_P95_ms'])/metric['E2E_P95_ms']))
            ids=sorted(blocks[0,'generic'][1])
            a=np.asarray([[1000*(blocks[r,'generic'][1][i]['finish_s']-blocks[r,'generic'][1][i]['arrival_s'])
                for i in ids] for r in range(3)])
            b=np.asarray([[1000*(blocks[r,'hindsight'][1][i]['finish_s']-blocks[r,'hindsight'][1][i]['arrival_s'])
                for i in ids] for r in range(3)])
            rng=np.random.default_rng(20261018)
            draws=[]
            for _ in range(2000):
                sample=rng.integers(0,c,c)
                repeats=rng.integers(0,3,3)
                aq=np.quantile(a[repeats][:,sample],.95,axis=1).mean()
                bq=np.quantile(b[repeats][:,sample],.95,axis=1).mean()
                draws.append(100*(1-bq/aq))
            agreement=[sum(blocks[r,'generic'][1][i]['output_ids']==blocks[r,'hindsight'][1][i]['output_ids'] for i in ids) for r in range(3)]
            repeat_parity={arm:[sum(blocks[0,arm][1][i]['output_ids']==blocks[r,arm][1][i]['output_ids'] for i in ids) for r in [1,2]]
                for arm in ['generic','hindsight']}
            prefix=[]
            for k in [32,64,128]:
                eligible=[i for i in ids if all(len(blocks[r,arm][1][i]['output_ids'])>=k for r in range(3) for arm in ['generic','hindsight'])]
                if not eligible:continue
                aq=np.mean([np.quantile([blocks[r,'generic'][1][i]['token_host_times_s'][k-1]*1000 for i in eligible],.95) for r in range(3)])
                bq=np.mean([np.quantile([blocks[r,'hindsight'][1][i]['token_host_times_s'][k-1]*1000 for i in eligible],.95) for r in range(3)])
                prefix.append(dict(K=k,n=len(eligible),P95_latency_to_K_gain_percent=100*(1-bq/aq),
                    scope='Intersection of naturally long outputs. The full concurrent workload still executes. No forced continuation.'))
            aq=np.quantile(a,.95,axis=1);bq=np.quantile(b,.95,axis=1)
            results.append(dict(C=c,repeats=3,requests=c,category='MEASURED_LIVE_HINDSIGHT_PLACEMENT',
                generic_P95_E2E_ms_by_repeat=aq.tolist(),hindsight_P95_E2E_ms_by_repeat=bq.tolist(),
                gain_percent=100*(1-bq.mean()/aq.mean()),CI95_gain_percent=np.quantile(draws,[.025,.975]).tolist(),
                exact_output_requests_between_arms=agreement,within_arm_repeat_output_parity=repeat_parity,
                generated_tokens={arm:[sum(o['generated_tokens'] for o in blocks[r,arm][1].values()) for r in range(3)]
                    for arm in ['generic','hindsight']},
                natural_prefix_controls=prefix,
                actual_route_model_validation=route_validation,
                quality='Unchanged gate/weights. Full benchmark quality under each placement NOT_RUN. Do not claim exact output parity between arms.',
                selection='Same exploratory requests used for oracle planning. Not untouched confirmation or a deployable policy.',
                bootstrap='2000 paired hierarchical draws over request IDs and three paired repeats. Limited repeat count.'))
    write(ROOT/'artifacts/live_hindsight_confirmation.json',dict(rows=results,metrics=detailed))
    print('LIVE_CONFIRMATION',results,flush=True)

if __name__=='__main__':main()
