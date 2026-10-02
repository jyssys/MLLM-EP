"""Reproduce the decisive locality contrast directly from published tensors."""
import argparse
import json
import numpy as np
from poc_agentic_ep.contract import ROOT
from poc_agentic_ep.metrics import coverage,clustered


def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args();root=ROOT/'results'/a.version
    manifest=json.loads((ROOT/'results/frozen_manifest_v2.json').read_text());lookup={r['request_id']:r for r in manifest['rows']}
    data=np.load(root/'aggregated.npz');ids=data['request_ids'].tolist();index={rid:i for i,rid in enumerate(ids)}
    counts=data['source_expert_counts'].sum(axis=3)
    calib=[i for i,rid in enumerate(ids) if lookup[rid]['split']=='calibration'];global_prefill=counts[calib,0].sum(axis=0)
    controls=json.loads((ROOT/'results/frozen_controls.json').read_text())['pairs'];deltas=[];groups=[]
    for pair in controls:
        target=counts[index[pair['target']],0];previous=counts[index[pair['previous']],0]
        deltas.append(float(coverage(previous,target,16).mean()-coverage(global_prefill,target,16).mean()))
        groups.append(pair['trajectory_id'])
    computed=clustered(deltas,groups);expected=json.loads((root/'locality_summary.json').read_text())['paired_deltas']['new_prefill']['adjacent_minus_global']['coverage_16']
    assert abs(computed['mean']-expected['mean'])<1e-12
    assert np.allclose(computed['ci95'],expected['ci95'],atol=1e-12)
    print(json.dumps(dict(status='PASS',previous_minus_global_coverage16_pp=100*computed['mean'],
        ci95_pp=[100*v for v in computed['ci95']],trajectories=computed['trajectories'],pairs=computed['pairs'])))


if __name__=='__main__':main()
