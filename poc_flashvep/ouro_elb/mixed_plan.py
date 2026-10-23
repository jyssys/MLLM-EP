"""A native Mixed baseline, separate from decode lease characterization."""
from .common import ROOT,write

def main():
    groups=[]
    for start,gpu in enumerate((0,4,7)):
        groups.append(dict(jobs=[dict(id=f'baseline_mixed_quiet_v1_s{start}_g{gpu}',gpu=gpu,kind='timing',
            mode='mixed',shapes=[[256,1],[2048,4],[2048,16]],groups=[1],graphs=True,**{'async':True},
            cap=64,repeats=3,lane='NATIVE_MIXED_QUIET_BASELINE',timeout_s=1800)]))
        groups.append(dict(jobs=[dict(id=f'baseline_fair_mixed_quiet_v1_s{start}_g{gpu}',gpu=gpu,kind='timing',
            mode='mixed',fairness='completion',shapes=[[256,1],[2048,4],[2048,16]],groups=[1],graphs=True,**{'async':True},
            cap=64,repeats=3,lane='FAIR_MIXED_QUIET_BASELINE',timeout_s=1800)]))
        groups.append(dict(jobs=[dict(id=f'baseline_fair_refill_quiet_v1_s{start}_g{gpu}',gpu=gpu,kind='timing',
            fairness='completion',shapes=[[256,1],[2048,16]],groups=[1,2,4],graphs=True,**{'async':True},
            cap=64,repeats=3,lane='FAIR_REFILL_QUIET_BASELINE',timeout_s=1800)]))
    write(ROOT/'mixed_baseline_plan.json',dict(gpus=list(range(8)),groups=groups))

if __name__=='__main__':main()
