"""Check Output200 context, state and lifetime before clean sweeps."""
import time
from .common import ROOT,save,write
from .prepare import qualification
from .qualification import gate
from .run import execute


def main():
    values=[]
    reference='qual200_Native_Last0_N8_v3'
    for method,layers in [('Native',0),('Agg',0)]+[(a,m) for m in (4,8,12) for a in ('Layer','Token')]:
        cfg=qualification(method,8,layers,version=10)
        cfg.update(name=f'qual200_{method}_Last{layers}_N8_v3',cap=200,
            request_caps=[200]*8,phase='UNIFIED_OUTPUT200_FULL_STATE_QUALIFICATION')
        assert execute(save(cfg)),cfg['name']
        values.append(gate(cfg['name'],reference))
    write(ROOT/'analysis/qualification_reserved_Output200_N8.json',dict(
        time=time.time(),passed=True,names=[r['name'] for r in values],
        evidence='CORRECTNESS_ONLY',scope='Full state and all200 output tokens. No latency claim.'))


if __name__=='__main__':
    main()
