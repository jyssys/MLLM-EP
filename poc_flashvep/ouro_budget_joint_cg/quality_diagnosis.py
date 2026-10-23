"""Report paired accuracy changes and retained baseline-correct losses."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT
from .cost_predictor import csv_file

def main():
    rows=list(csv.DictReader((ROOT/'quality_cases.csv').open()))
    by={(r['policy'],r['id']):r for r in rows};losses=[];direct=[]
    for row in rows:
        ref=by[('F0A',row['id'])]
        if ref['correct']=='True' and row['correct']!='True':
            losses.append(dict(dataset=row['dataset'],id=row['id'],candidate=row['policy'],
                reference='F0A',gold=row['gold'],reference_extracted=ref['extracted'],candidate_extracted=row['extracted'],
                candidate_truncated=row['truncated'],reference_file=ref['file'],candidate_file=row['file']))
    rng=np.random.default_rng(20261023)
    for domain in ('gsm8k','math500'):
        joint=[r for r in rows if r['policy']=='J1' and r['dataset']==domain]
        delta=[];exact=0
        for row in joint:
            ref=by[('G1',row['id'])]
            delta.append(100*(int(row['correct']=='True')-int(ref['correct']=='True')))
            a=json.loads((ROOT/row['file']).read_text());b=json.loads((ROOT/ref['file']).read_text())
            exact+=a['output_ids']==b['output_ids']
        delta=np.array(delta);draws=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
        direct.append(dict(dataset=domain,N=len(joint),candidate='J1',reference='G1',
            accuracy_delta_pp=float(delta.mean()),CI95_low=float(np.percentile(draws,2.5)),CI95_high=float(np.percentile(draws,97.5)),
            token_exact=exact,identical_correctness_labels=int(sum(value==0 for value in delta)),
            scope='Matched public quality pilot. No population equivalence claim.'))
    csv_file(ROOT/'quality_baseline_correct_to_wrong.csv',losses)
    csv_file(ROOT/'quality_J1_vs_G1.csv',direct)
    print(json.dumps(dict(baseline_correct_to_wrong_cases=len(losses),direct=direct),indent=2))

if __name__=='__main__':
    main()
