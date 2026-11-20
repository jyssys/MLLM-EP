"""Unroll the measured FIFO destination chain, not a queueing approximation."""
import json
import numpy as np
from .common import ROOT,PREVIOUS,write
from .historical_bridge import HISTORICAL,SEEDS
from .queue_accounting import lines


def decompose(rows):
    n=len(rows);assert n>1
    origin=rows[0]['arrival'];prepared=[];decode=[];delivery=[];supply=[];host=[]
    for i,row in enumerate(rows):
        assert row['ready']<=row['D_submit']<=row['first_worker']<=row['last_worker']<=row['finish']
        prepared.append(row['first_worker']-row['D_submit'])
        decode.append(row['last_worker']-row['first_worker'])
        delivery.append(row['finish']-row['last_worker'])
        if i:
            assert row['D_submit']>=rows[i-1]['finish']-1e-6
            supply.append(max(0.,row['ready']-rows[i-1]['finish']))
            host.append(max(0.,row['D_submit']-max(row['ready'],rows[i-1]['finish'])))
    initial=rows[0]['D_submit']-origin
    parts=dict(initial_D_submit=1000*initial,
        weighted_prior_D_preparation=1000*sum((n-1-j)/n*prepared[j] for j in range(n-1)),
        weighted_prior_decode_residency=1000*sum((n-1-j)/n*decode[j] for j in range(n-1)),
        weighted_prior_delivery=1000*sum((n-1-j)/n*delivery[j] for j in range(n-1)),
        weighted_D_waiting_for_ready_supply=1000*sum((n-j)/n*supply[j-1] for j in range(1,n)),
        weighted_eligible_destination_host_gap=1000*sum((n-j)/n*host[j-1] for j in range(1,n)))
    measured=1000*float(np.mean([r['D_submit']-origin for r in rows]))
    assert abs(sum(parts.values())-measured)<1e-3,(parts,measured)
    return dict(N=n,mean_D_submit_relative_to_first_arrival_ms=measured,exact_parts_ms=parts,
        mean_ready_relative_to_first_arrival_ms=1000*float(np.mean([r['ready']-origin for r in rows])),
        mean_ready_D_wait_ms=1000*float(np.mean([r['D_submit']-r['ready'] for r in rows])),
        mean_D_preparation_ms=1000*float(np.mean(prepared)),
        mean_decode_residency_ms=1000*float(np.mean(decode)),
        prior_completion_transitions=n-1,
        ready_supply_already_available_at_prior_client_completion=sum(r['ready']<=p['finish'] for p,r in zip(rows,rows[1:])),
        ready_supply_missing_fraction=float(np.mean(np.asarray(supply)>0)),
        mean_host_D_wait_for_P_ready_ms=1000*float(np.mean(supply)),
        mean_eligible_host_gap_ms=1000*float(np.mean(host)),
        evidence='TRACE_ANALYSIS',scope='Exact unrolling of a measured FIFO whole-request host chain. '
        'Weighted service differences are not randomized independent causal effects. D waiting for P-ready supply is not physical GPU idle or a decode-safe window.')


def historical_rows(axis,seed):
    cut=0 if axis=='Native' else 8
    name=f'heldout_confirmation_{axis}_Last{cut}_N72_O16_q6_s{seed}_B1_W32_v1'
    host=json.loads((HISTORICAL/'analysis'/(name+'_host_paths.json')).read_text())
    requests={f'q{r["sample_id"]}':r for r in lines(HISTORICAL/'raw'/name/'requests.jsonl')}
    paths=sorted(host['request_paths'],key=lambda r:int(r['request_id'][1:]));rows=[]
    for path in paths:
        rid=path['request_id'];r=requests[rid];bounds=path['boundaries_monotonic']
        rows.append(dict(arrival=bounds[0],ready=bounds[2],D_submit=bounds[3],
            first_worker=r['token_events'][0]['worker_time'],last_worker=r['token_events'][-1]['worker_time'],finish=r['finish']))
    return name,rows


def current_rows(name,root=ROOT):
    folder=root/'raw'/name
    handoff={r['request_id']:r for r in lines(folder/'handoffs.jsonl')}
    submissions={(r['role'],r['request_id']):r['time'] for r in lines(folder/'request_submissions.jsonl')}
    requests=sorted(lines(folder/'requests.jsonl'),key=lambda r:r['sample_id']);rows=[]
    for r in requests:
        rid=f'q{r["sample_id"]}'
        rows.append(dict(arrival=r['arrival'],ready=handoff[rid]['received'],D_submit=submissions['decode',rid],
            first_worker=r['token_events'][0]['worker_time'],last_worker=r['token_events'][-1]['worker_time'],finish=r['finish']))
    return rows


def main():
    old=[]
    for seed in SEEDS:
        for axis in ('Native','Layer','Token'):
            name,rows=historical_rows(axis,seed)
            old.append(dict(name=name,axis=axis,seed=seed,**decompose(rows)))
    differences={key:float(np.mean([r['exact_parts_ms'][key] for r in old if r['axis']=='Layer'])-
        np.mean([r['exact_parts_ms'][key] for r in old if r['axis']=='Token'])) for key in old[0]['exact_parts_ms']}
    new=[]
    source=ROOT/'analysis/historical_runtime_bridge_v1.json'
    if source.exists():
        for r in json.loads(source.read_text())['results']:
            if r['status'] in ('MEASURED_ONLINE','DIAGNOSTIC'):
                new.append(dict(name=r['name'],axis=r['axis'],BI=r['BI'],seed=r['seed'],**decompose(current_rows(r['name']))))
    longbench=[]
    for axis in ('Layer','Token'):
        name=f'NaturalLast4Repro_{axis}_clean_before_q2.35_c4_s2026112061_N200_v1'
        longbench.append(dict(name=name,axis=axis,**decompose(current_rows(name))))
    write(ROOT/'analysis/exact_destination_chain_decomposition_v1.json',dict(historical=old,
        historical_Layer_minus_Token_exact_weighted_parts_ms=differences,current_historical_workload=new,
        current_LongBench_Last4_Natural=longbench,
        scope='Separate cut/workload families. Exact accounting, not an ACK removal, GPU-only intervention or a future-aware online predictor.'))
    print(json.dumps(dict(historical_weighted_parts=differences,
        current_LongBench=[{k:r[k] for k in ('axis','ready_supply_missing_fraction','mean_host_D_wait_for_P_ready_ms')} for r in longbench]),indent=2))


if __name__=='__main__':main()
