"""Update both FIFO servers and source lifetime credit after each action.

This model applies only to the explicit P1/D-prefill1/D-decode1 protocol.
It does not model ordinary continuous batching or active-decode co-execution.
An async caller must include post-EOS GPU drain in D service or D control.
Client last-token time alone does not imply device-wide quiescence.
"""
import json
import numpy as np
from .common import ROOT,write
from .qualification import records


def replay(arrivals,P_services,D_first_services,D_decode_services,credit=4,P_control=None,D_control=None):
    n=len(arrivals)
    assert credit>=1 and all(len(values)==n for values in (P_services,D_first_services,D_decode_services))
    P_control=[0.]*n if P_control is None else P_control
    D_control=[0.]*n if D_control is None else D_control
    output=[]
    for index in range(n):
        assert min(P_services[index],D_first_services[index],D_decode_services[index],P_control[index],D_control[index])>=0
        previous=output[index-1] if index else None
        release=output[index-credit]['D_first'] if index>=credit else 0.
        P_start=max(arrivals[index],previous['P_finish'] if previous else 0.,release)+P_control[index]
        P_finish=P_start+P_services[index]
        D_start=max(P_finish,previous['D_finish'] if previous else 0.)+D_control[index]
        D_first=D_start+D_first_services[index];D_finish=D_first+D_decode_services[index]
        output.append(dict(index=index,P_start=P_start,P_finish=P_finish,D_start=D_start,D_first=D_first,D_finish=D_finish,
            TTFT=D_first-arrivals[index],P_queue=P_start-arrivals[index],D_ready_wait=D_start-P_finish,
            source_credit_release=release))
    return output


def structural_check(name):
    folder=ROOT/'raw'/name;cfg=json.loads((folder/'configuration.json').read_text())
    assert cfg['controlled_pipeline'] and cfg['method']=='Native' and not cfg.get('axis_kind')
    origin=json.loads((folder/'window.json').read_text())['start']
    decisions={r['request_id']:r for r in records(folder/'axis_decisions.jsonl')}
    hands={r['request_id']:r for r in records(folder/'handoffs.jsonl')}
    data=sorted(records(folder/'requests.jsonl'),key=lambda r:r['sample_id'])
    admissions=records(folder/'admissions.jsonl')
    arrivals=[];P=[];F=[];D=[];pc=[];dc=[];truth=[]
    credit=cfg['source_credit_limit']
    for index,row in enumerate(data):
        rid=f'q{index}';ps=decisions[rid]['time']-origin;pf=hands[rid]['received']-origin
        ds=next(r['worker_accepted'] for r in admissions if r['request_id']==rid and r['role']=='decode')-origin
        first=row['first_token']-origin;end=row['finish']-origin;arrival=row['arrival']-origin
        pbase=max(arrival,truth[-1]['P_finish'] if truth else 0.,truth[index-credit]['D_first'] if index>=credit else 0.)
        dbase=max(pf,truth[-1]['D_finish'] if truth else 0.)
        pc.append(ps-pbase);dc.append(ds-dbase)
        assert pc[-1]>=-1e-6 and dc[-1]>=-1e-6,(index,pc[-1],dc[-1])
        pc[-1]=max(pc[-1],0.);dc[-1]=max(dc[-1],0.)
        arrivals.append(arrival);P.append(pf-ps);F.append(first-ds);D.append(end-first)
        truth.append(dict(P_start=ps,P_finish=pf,D_start=ds,D_first=first,D_finish=end))
    predicted=replay(arrivals,P,F,D,credit,pc,dc)
    errors={key:float(max(abs(a[key]-b[key]) for a,b in zip(predicted,truth))) for key in truth[0]}
    passed=all(value<1e-5 for value in errors.values())
    write(ROOT/'analysis'/(name+'_tandem_structure_check.json'),dict(passed=passed,max_error_seconds=errors,
        source_credit=credit,P_control_ms=[1000*x for x in pc],D_control_ms=[1000*x for x in dc],
        observed_services=dict(P=P,D_first=F,D_decode=D),arrivals=arrivals,
        evidence='TRACE_ANALYSIS',
        scope='Observed services and observed per-request control delays reconstruct the same trace. '
        'This tests the state transition, not predictive accuracy or oracle headroom. Independent cost fit and held-out online validation are still required.'))
    return passed


if __name__=='__main__':
    import sys
    print(structural_check(sys.argv[1]))
