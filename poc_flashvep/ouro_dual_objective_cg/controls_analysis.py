"""Summarize measured controls without promoting service sums to E2E shares."""
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,write
from ..ouro_budget_joint_cg.cost_predictor import csv_file

def main():
    rows=[];probes=[]
    for marker in sorted((ROOT/'raw').glob('dual_p0_*/completed.json')):
        folder=marker.parent;job=json.loads((folder/'identity.json').read_text())['job']
        for file in folder.glob('cal_*.json'):
            data=json.loads(file.read_text())
            rows.append(dict(GPU=job['gpu'],workers=job['workers'],start=job['repeat'],
                P=data['P_bin'],N=data['N'],rep=data['repetition'],median_ITL_ms=data['median_ITL_ms'],
                output_tokens_s=data['drained_output_tokens_s'],source=str(file.relative_to(ROOT))))
    grouped=defaultdict(list)
    for r in rows:grouped[(r['GPU'],r['workers'],r['start'],r['P'])].append(r)
    drift=[]
    for (gpu,workers,start,P),values in grouped.items():
        if workers==1:continue
        base=grouped[(gpu,1,start,P)]
        drift.append(dict(GPU=gpu,workers=workers,start=start,P=P,
            median_ITL_drift_percent=100*(np.median([r['median_ITL_ms'] for r in values])/np.median([r['median_ITL_ms'] for r in base])-1),
            throughput_drift_percent=100*(np.median([r['output_tokens_s'] for r in values])/np.median([r['output_tokens_s'] for r in base])-1),
            evidence='LIVE_MEASURED_CONTENTION_CALIBRATION_WITH_UNTOUCHED_GPU0_3_BACKGROUND'))
    for marker in sorted((ROOT/'raw').glob('dual_p1_*/completed.json')):
        folder=marker.parent
        for file in folder.glob('P*.json'):
            data=json.loads(file.read_text())
            if 'comparison' not in data:continue
            comp=data['comparison']
            probes.append(dict(source=str(file.relative_to(ROOT)),P=data['P'],mode=data['mode'],
                token_exact=comp['tokens_exact'],hidden_exact=comp['hidden']['exact'],logits_exact=comp['logits']['exact'],
                KV_exact=comp['KV_exact'],tolerance_pass=data['numerical_tolerance_pass'],
                full_R4=data['exact_prompt_work']==data['expected_prompt_work']))
    csv_file(ROOT/'contention_raw.csv',rows);csv_file(ROOT/'contention_paired_drift.csv',drift)
    csv_file(ROOT/'numerical_probe_results.csv',probes)
    review=any(abs(r['throughput_drift_percent'])>3 or abs(r['median_ITL_drift_percent'])>3 for r in drift)
    summary=dict(calibration_cohorts=len(rows),independent_calibration_starts=len(list((ROOT/'raw').glob('dual_p0_*/completed.json'))),
        numerical_comparisons=len(probes),all_numerical_pass=bool(probes) and all(r['tolerance_pass'] and r['full_R4'] for r in probes),
        all_probe_tokens_exact=bool(probes) and all(r['token_exact'] for r in probes),
        review_required=review,diagnostic_workers=4,final_performance_workers=1,
        final_scope='One measurement worker, fixed untouched0–3 background. Not a fully quiescent eight-device host.',
        notes='Three repeats per start are not independent process starts. Two starts per GPU/worker-count cell.')
    write(ROOT/'controls_summary.json',summary)
    print(json.dumps(summary,indent=2));print('Throughput drift extrema',
        min(r['throughput_drift_percent'] for r in drift),max(r['throughput_drift_percent'] for r in drift))

if __name__=='__main__':main()
