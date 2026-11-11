"""Require whole-run and aligned local-tail controls for each observer."""
import argparse,json
from .common import *
from .analyze import native,physical,local_itl
from .gates import interference


def analyze(plan):
    initialize();plan=Path(plan);cfg=json.loads(plan.read_text())
    clean=cfg['clean_reference'];repeat=cfg['jobs'][-1]
    names=[clean,*cfg['jobs']];metrics={n:native(n,clean) for n in names}
    assert all(m['complete'] for m in metrics.values()),'Finish all controls before qualification'
    fields=('response_wall_seconds','TTFT_p95_ms','TPOT_p95_ms','ITL_p99_ms','maximum_ITL_ms')
    controls=[clean,repeat];aa=interference(metrics[repeat],metrics[clean],fields);out=[]
    for name in cfg['jobs'][:-1]:
        config=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        whole={c:interference(metrics[name],metrics[c],fields) for c in controls}
        trace=physical(name) if config.get('discovery_activity') else None
        local={};local_gate={}
        if trace:
            origin=json.loads((ROOT/'raw'/name/'measurement_begin.json').read_text())['time']
            start=trace['decode']['left']-origin;end=trace['decode']['right']-origin
            local={n:local_itl(n,start,end) for n in [*controls,name]}
            local_gate={c:interference(local[name],local[c],('p99','maximum'))
                if min(local[name]['count'],local[c]['count'])>=20 else
                dict(passed=False,reason='Fewer than 20 local intervals') for c in controls}
        valid=all(m['correctness_passed'] for m in metrics.values())
        qualified=valid and aa['passed'] and all(s['passed'] for s in whole.values())
        if trace:qualified &= all(s['passed'] for s in local_gate.values()) and all(v['collector_valid'] for v in trace.values())
        out.append(dict(name=name,whole_run_screens=whole,local_screens=local_gate,local_ITL=local,
            trace=trace,qualified=qualified,status='QUALIFIED_OBSERVER_PILOT' if qualified else 'DIAGNOSTIC_ONLY',
            Phase_1_qualified=False,
            scope='Observer point screen only. Independent starts, capacity, queue ownership, and source/handoff feasibility remain separate gates.'))
    value=dict(plan=str(plan),metrics=metrics,Native_AA=aa,observers=out)
    write(ROOT/'analysis'/(plan.stem+'_results.json'),value)
    print(json.dumps(dict(Native_AA=aa,observers=[{k:v for k,v in r.items() if k in ('name','status','whole_run_screens','local_screens')} for r in out])))
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('plan');analyze(p.parse_args().plan)
