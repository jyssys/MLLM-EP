"""Reserve three starts for primary points and retain exploratory endpoints."""
import fcntl
import time
from .common import ROOT,QUIET_LOCK,resources,save,write
from .heldout_baselines import configuration,SEEDS,RATES
from .analysis import metrics
from .run import execute


def main():
    original=[(cap,qps,seed,method) for cap in (16,64) for qps in RATES
              for seed in SEEDS for method in ('Native','Colocated')]
    selected=[]
    for cap,qps in ((16,6.4),(64,4.8),(64,6.4),(16,2.),(16,10.),(16,4.8),(64,2.),(64,10.)):
        seeds=SEEDS if (cap,qps) in ((16,6.4),(16,2.),(64,4.8),(64,6.4)) else SEEDS[:1]
        for seed in seeds:
            methods=('Native','Colocated') if seed%2==0 else ('Colocated','Native')
            selected.extend((cap,qps,seed,method) for method in methods)
    skipped=[configuration(method,cap,qps,seed) for cap,qps,seed,method in original
             if (cap,qps,seed,method) not in selected]
    for cfg in skipped:
        save(cfg)
        write(ROOT/'state/not_run_budget_scope_v2'/(cfg['name']+'.json'),dict(
            status='NOT_RUN',reason='Exploratory endpoint replication reduced before candidate confirmation. No method, trace, precision or SLO threshold changed.'))
    write(ROOT/'state/evaluation_budget_revision_v2.json',dict(time=time.time(),
        original_baseline_starts=len(original),planned_baseline_starts=len(selected),
        retained_three_start_points=[[16,6.4],[16,2.],[64,4.8],[64,6.4]],
        exploratory_endpoint_starts=1,skipped=[c['name'] for c in skipped],
        reason='Prioritize matched layer/token and dynamic/static independent comparisons within the10-hour research budget. The candidate has not been selected or evaluated on held-out latency.',
        unchanged=['model','SLO','dataset','arrival traces','GPU pair','execution settings']))
    values=[]
    try:
        for cap,qps,seed,method in selected:
            cfg=configuration(method,cap,qps,seed)
            if not execute(save(cfg)):
                continue
            value=metrics(cfg['name']);assert value['output_gate']['passed']
            values.append(value);write(ROOT/'analysis/heldout_baselines_budget_v2.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
