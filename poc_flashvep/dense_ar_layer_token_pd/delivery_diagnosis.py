"""Retain client SLO failures while separating worker and delivery intervals."""
import collections
import json
from .common import ROOT,write
from .analysis import records


def main():
    limits=json.loads((ROOT/'configs/frozen_SLO_regimes.json').read_text())['output_caps']
    runs=[]
    for path in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        if not json.loads(path.read_text()).get('complete'):
            continue
        cfg=json.loads((path.parent/'configuration.json').read_text())
        if cfg.get('correctness_only') or not (cfg.get('flow_balance_policy') or cfg.get('phase','').startswith('FROZEN_HELDOUT')):
            continue
        counts=collections.Counter();samples=[]
        for row in records(path.parent/'requests.jsonl'):
            events=row['token_events'];cap=row['output_tokens'];bound=limits[str(cap)]['Decode_priority']
            pairs=list(zip(events,events[1:]));parent=[1000*(b['time']-a['time']) for a,b in pairs]
            worker=[1000*(b['worker_time']-a['worker_time']) for a,b in pairs]
            failTTFT=row['ttft_ms']>bound['TTFT_ms'];failTPOT=row['tpot_ms']>bound['TPOT_ms']
            failP=max(parent)>bound['ITL_ms'];failW=max(worker)>bound['ITL_ms']
            counts['requests']+=1;counts['TTFT_failures']+=failTTFT;counts['TPOT_failures']+=failTPOT
            counts['client_max_ITL_failures']+=failP;counts['worker_max_interval_exceedances']+=failW
            counts['client_ITL_failure_with_worker_intervals_below_limit']+=failP and not failW
            counts['client_joint_failures']+=failTTFT or failTPOT or failP
            if failP and not failW:
                counts['delivery_only_ITL_failure_Output'+str(cap)]+=1
                for (a,b),p,w in zip(pairs,parent,worker):
                    if p<=bound['ITL_ms']:
                        continue
                    delta=1000*((b['time']-b['worker_time'])-(a['time']-a['worker_time']))
                    assert abs(p-w-delta)<1e-5
                    samples.append(dict(sample_id=row['sample_id'],cap=cap,client_interval_ms=p,
                        worker_interval_ms=w,delivery_lag_change_ms=delta,
                        previous_delivery=a['time'],delivery=b['time'],client_SLO='FAILED_UNCHANGED'))
        runs.append(dict(name=path.parent.name,counts=dict(counts),delivery_only_ITL_examples=samples,
            evidence='TRACE_ANALYSIS',scope='The client SLO stays unchanged. Worker intervals still include worker host scheduling and GPU work. This is an exact interval identity, not exclusive kernel timing, not additive p99 attribution and not proof that a specific controller call caused delivery delay. Request failure counts can overlap.'))
    write(ROOT/'analysis/worker_delivery_SLO_failure_diagnosis.json',dict(runs=runs,
        evidence='TRACE_ANALYSIS',scope='Use the frozen client/coordinator metric for all main results. Never relabel a client failure as a pass by changing the measurement boundary.'))
    print('WORKER_DELIVERY_DIAGNOSIS',len(runs),flush=True)


if __name__=='__main__':
    main()
