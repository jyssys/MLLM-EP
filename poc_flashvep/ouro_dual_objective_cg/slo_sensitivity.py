"""Frozen secondary SLO thresholds on the same observed output clocks."""
import csv
from collections import defaultdict
import numpy as np
from .common import ROOT,write
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    rows=[r for r in csv.DictReader((ROOT/'serving_results.csv').open()) if r['split']=='FINAL']
    lookup={(r['run_id'],r['scenario'],r['policy']):r for r in rows};groups=defaultdict(list)
    for left in rows:
        if left['policy']!='O6_J':continue
        for control in ('O6_G','O6_G1024','O6_C','BEST_FIXED','MIXED_STOCK','MIXED_FAIR'):
            right=lookup.get((left['run_id'],left['scenario'],control))
            if right is None:continue
            for ttft in (2,5,10):
                for tbt in (100,150,200,250,500):
                    metric=f'drain_goodput_TTFT{ttft}s_TBT{tbt}ms_rps'
                    key=(left['lane'],left['workload'],left['offered_rate_rps'],control,ttft,tbt)
                    groups[key].append((left['run_id'],float(left[metric]),float(right[metric])))
    rng=np.random.default_rng(2026102322);out=[]
    for key,values in groups.items():
        lane,workload,rate,control,ttft,tbt=key;blocks=defaultdict(list)
        for run,a,b in values:blocks[run].append((a,b))
        b=np.array([np.mean(v,axis=0) for v in blocks.values()]);boot=b[rng.integers(0,len(b),size=(3000,len(b)))].mean(axis=1)
        d=boot[:,0]-boot[:,1]
        out.append(dict(lane=lane,workload=workload,rate=rate,candidate='O6_J',reference=control,
            TTFT_s=ttft,TBT_ms=tbt,independent_starts=len(b),candidate_goodput_rps=float(b[:,0].mean()),
            reference_goodput_rps=float(b[:,1].mean()),ratio=None if b[:,1].mean()==0 else float(b[:,0].mean()/b[:,1].mean()),
            delta_rps=float((b[:,0]-b[:,1]).mean()),CI95_low=float(np.percentile(d,2.5)),CI95_high=float(np.percentile(d,97.5)),
            scope='LIVE_MEASURED_POSTPROCESS_FROZEN_SECONDARY_GRID. Pointwise process-block CI, not multiplicity-adjusted. No controller retuning or capacity qualification.'))
    csv_file(ROOT/'SLO_sensitivity_paired.csv',out)
    write(ROOT/'SLO_sensitivity_scope.json',dict(rows=len(out),TTFT_s=[2,5,10],TBT_ms=[100,150,200,250,500],
        primary=(5,200),changes_policy=False,secondary_results_do_not_replace_primary_gate=True))
    print('Frozen SLO contrasts',len(out))

if __name__=='__main__':main()
