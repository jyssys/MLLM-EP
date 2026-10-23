"""Freeze one primary objective using VALIDATION and fixed controls using TRAIN."""
import csv
import json
from collections import defaultdict
import numpy as np
from .common import ROOT,write,digest

def mean(rows,key):return float(np.mean([float(r[key]) for r in rows]))

def main():
    assert not (ROOT/'objective_shortlist.json').exists(),'Preserve the frozen final choice'
    for name in ('p2_objective_validation_plan.json','p2_fixed_tuning_plan.json'):
        plan=json.loads((ROOT/name).read_text())
        assert all((ROOT/'raw'/j['id']/'completed.json').exists() for j in plan['jobs']),name
    rows=list(csv.DictReader((ROOT/'serving_results.csv').open()))
    val=[r for r in rows if r['split']=='VALIDATION']
    metric='drain_goodput_TTFT5s_TBT200ms_rps';families=[]
    for family in ('O2','O4','O6','O10'):
        candidate=[r for r in val if r['policy']==family+'_J']
        controls={label:[r for r in val if r['policy']==label] for label in (family+'_G',family+'_G1024')}
        scores=[]
        for width in ('4','8'):
            c=[r for r in candidate if r['width']==width]
            strong=max(mean([r for r in group if r['width']==width],metric) for group in controls.values())
            score=mean(c,metric)/strong if strong else 0.
            scores.append(score)
        families.append(dict(family=family,worst_stratum_goodput_ratio=min(scores),stratum_ratios=scores,
            fresh_C_change_fraction=sum(int(r['C_differs_fixed_same_state']) for r in candidate)/max(1,sum(int(r['fresh_calls']) for r in candidate))))
    # Locked rule: maximize the worst of the two validation strata against the
    # stronger fixed-C G-only control. No FINAL data or accuracy label selects it.
    chosen=max(families,key=lambda r:(r['worst_stratum_goodput_ratio'],r['family']=='O2'))
    development=json.loads((ROOT/'objective_shortlist_development.json').read_text())
    selected=next(c for c in development['arms'] if c['label']==chosen['family']+'_J')
    train=[r for r in rows if r['split']=='TRAIN']
    groups=defaultdict(list)
    for row in train:groups[row['policy']].append(row)
    best=max(groups,key=lambda label:(mean(groups[label],metric),-mean(groups[label],'ITL_P99_ms'),-mean(groups[label],'TTFT_P95_ms')))
    mixed_labels=[label for label in groups if 'MIXED' in label]
    best_mixed=max(mixed_labels,key=lambda label:mean(groups[label],metric))
    plan=json.loads((ROOT/'p2_fixed_tuning_plan.json').read_text())
    dictionary={c['label']:c for j in plan['jobs'] for c in j['cells']}
    output=dict(primary_objective=selected,validation_candidates=families,
        selection_rule='Maximum worst-stratum VALIDATION goodput ratio against stronger G2048/G1024. No FINAL tuning.',
        best_tested_fixed=dictionary[best],best_tested_mixed=dictionary[best_mixed],
        fixed_tuning='Fresh TRAIN, C128/512/1024/2048 x all four groups, plus Atomic and Mixed. Not a global oracle.',
        primary_contrast='Selected Joint vs both same-objective fixed-C G-only controls. C-only and native Mixed remain mandatory.',
        provisional_incremental_gate=chosen['worst_stratum_goodput_ratio']>=1.05 and chosen['fresh_C_change_fraction']>=.05,
        qualification='VALIDATION selection only. No new quality or qualified-capacity claim.',
        serving_results_sha256=digest(ROOT/'serving_results.csv'))
    write(ROOT/'objective_shortlist.json',output)
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
