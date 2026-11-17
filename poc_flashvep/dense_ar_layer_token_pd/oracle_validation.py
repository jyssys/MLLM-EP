"""Compare the modeled Joint95 frontier without changing frozen costs."""
import fcntl
import json
import time
from .common import ROOT,QUIET_LOCK,resources,write,digest
from .replay import load_model,groups,oracle,evaluate
from .strong_static import frontier


def main():
    target=ROOT/'analysis/strong_per_cap_fraction_oracle_v2_Joint95.json'
    assert not target.exists(),'Preserve a frozen model-analysis result'
    screen=json.loads((ROOT/'analysis/flow_macro8_oracle_screen_v3.json').read_text())
    selected=next(r for r in screen if r['rate']==4.8)
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not resources().processes()
        resources().restore(list(range(8)))
        started=time.time();model=load_model(8,'v3')
        workload=groups(selected['arrivals'],selected['caps'],8)
        strong=frontier(model,workload);beams=[]
        for width in (256,1024):
            state=oracle(model,workload,width=width)
            beams.append(dict(width=width,metric=evaluate(state),plan=[d['layers'] for d in state.decisions]))
        valid=[r for r in beams if r['metric']['Joint_attainment']>=.95]
        best=max(valid,key=lambda r:r['metric']['SLO_output_tokens_s']) if valid else None
        static95=strong['best_with_Joint95']
        write(target,dict(time=time.time(),elapsed_s=time.time()-started,static=strong,beams=beams,
            best_tested_dynamic_with_Joint95=best,
            ratio_Joint95=best['metric']['SLO_output_tokens_s']/static95['metric']['SLO_output_tokens_s']
                if best and static95 else None,
            model_SHA256=digest(ROOT/'configs/empirical_host_cost_model_discovery_v3.json'),
            evidence='OFFLINE_ORACLE',scope='Repeat the same frozen mixed-cap replay. Keep the unconstrained token-rate maximum separate from the Joint95-constrained maximum. Beam256/1024 are approximate searches, not optimality bounds. No measured serving or sustainable-capacity claim.'))
        print('ORACLE_JOINT95',static95,beams,flush=True)


if __name__=='__main__':
    main()
