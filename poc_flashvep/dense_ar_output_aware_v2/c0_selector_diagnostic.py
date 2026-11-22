"""Test observable fixed-output rules without claiming an online scheduler."""
import argparse
import json
import time
import numpy as np
from .common import ROOT,write,digest
from .c0_analysis import csv_save

OUTPUTS=(8,16,64,128,192)
AXES=('Layer','Token')

def predict(rule,row):
    if rule['kind']=='fixed':return rule['axis']
    if rule['kind']=='output_threshold':
        return rule['low'] if row['output']<=rule['threshold'] else rule['high']
    assert rule['kind']=='input_then_output'
    return predict(rule['low'] if row['input_bucket']<=rule['input_threshold'] else rule['high'],row)

def loss(rule,rows):
    return float(np.mean([r[predict(rule,r)+'_TTFT_p95_ms']/r['Native_TTFT_p95_ms'] for r in rows]))

def candidates():
    yield from (dict(kind='fixed',axis=axis,leaves=1) for axis in AXES)
    for threshold in OUTPUTS[:-1]:
        for low,high in (('Layer','Token'),('Token','Layer')):
            yield dict(kind='output_threshold',threshold=threshold,low=low,high=high,leaves=2)

def best(rows):
    return min(candidates(),key=lambda r:(loss(r,rows),r['leaves'],json.dumps(r,sort_keys=True)))

def freeze():
    path=ROOT/'configs/c0_output_selector_rules_frozen_v2.json';assert not path.exists()
    source=ROOT/'analysis/c0_output_map_v2.json';data=json.loads(source.read_text())
    rows=[r for r in data['cells'] if r['split']=='discovery'];assert len(rows)==80 and all(r['status']=='COMPLETE' for r in rows)
    rules={}
    for cut in (2,4,8,12):
        group=[r for r in rows if r['cut']==cut];fixed=min((r for r in candidates() if r['kind']=='fixed'),key=lambda r:loss(r,group))
        output=best(group);options=[output]
        for threshold in (0,1,2):
            low=best([r for r in group if r['input_bucket']<=threshold]);high=best([r for r in group if r['input_bucket']>threshold])
            options.append(dict(kind='input_then_output',input_threshold=threshold,low=low,high=high,leaves=low['leaves']+high['leaves']))
        two=min(options,key=lambda r:(loss(r,group),r['leaves'],json.dumps(r,sort_keys=True)))
        rules[str(cut)]={label:dict(rule=rule,discovery_loss=loss(rule,group)) for label,rule in
            (('FixedAxis',fixed),('OutputOnly',output),('InputAndOutput',two))}
    write(path,dict(time=time.time(),discovery_source_SHA256=digest(source),training_cells=80,heldout_cells_used=0,rules=rules,
        features=['known controlled max_tokens','known input-token bucket'],action_metadata='Cut is a fixed calibrated action,not a predicted feature.',
        fitting_loss='Mean TTFT p95 divided by matched C0 Native TTFT p95 across discovery cells.',
        scope='Diagnostic rule learning from N1 discovery cells only. C0 uses a Mode0 Token,not strong C3 Stock Token. '
            'A rule chooses one static axis for an entire homogeneous controlled-output trace. '
            'It does not alternate per-request actions or simulate queue updates. '
            'Do not replace known controlled max_tokens with future natural EOS. '
            'This experiment cannot establish dynamic C3 serving gain or a mixed-workload oracle upper bound.'))
    print('C0_OBSERVABLE_RULES_FROZEN',rules,flush=True)

def evaluate():
    frozen=json.loads((ROOT/'configs/c0_output_selector_rules_frozen_v2.json').read_text())
    data=json.loads((ROOT/'analysis/c0_output_map_v2.json').read_text())
    cells=[r for r in data['cells'] if r['split']=='heldout' and r['completed_paired_starts']==3]
    records=[]
    for row in cells:
        for label,item in frozen['rules'][str(row['cut'])].items():
            axis=predict(item['rule'],row);winner='Layer' if row['TTFT_ratio']<1 else 'Token' if row['TTFT_ratio']>1 else 'EQUAL'
            selected=row[axis+'_TTFT_p95_ms'];minimum=min(row[a+'_TTFT_p95_ms'] for a in AXES)
            records.append(dict(model=label,input_bucket=row['input_bucket'],output=row['output'],cut=row['cut'],
                selected_axis=axis,point_winner=winner,prediction_correct=axis==winner or winner=='EQUAL',
                TTFT_p95_ratio=row['TTFT_ratio'],TTFT_CI95=row['TTFT_CI95'],
                statistically_confirmed=row['statistically_confirmed'],both_arms_decode_qualified=row['both_arms_controlled_decode_gate_pass'],
                selected_TTFT_p95_ms=selected,better_measured_fixed_axis_TTFT_p95_ms=minimum,
                TTFT_p95_regret_ms=selected-minimum,TTFT_p95_regret_fraction=selected/minimum-1,
                evidence='OFFLINE_PREDICTION_DIAGNOSTIC'))
    summary=[]
    for label in ('FixedAxis','OutputOnly','InputAndOutput'):
        for scope in ('All completed cells','Statistically confirmed cells','Both arms decode-qualified cells'):
            rows=[r for r in records if r['model']==label and (scope=='All completed cells' or
                scope=='Statistically confirmed cells' and r['statistically_confirmed'] in ('LAYER_STATISTICALLY_CONFIRMED','TOKEN_STATISTICALLY_CONFIRMED') or
                scope=='Both arms decode-qualified cells' and r['both_arms_decode_qualified'])]
            summary.append(dict(model=label,scope=scope,cells=len(rows),point_winner_accuracy=float(np.mean([r['prediction_correct'] for r in rows])) if rows else None,
                mean_cell_TTFT_p95_regret_ms=float(np.mean([r['TTFT_p95_regret_ms'] for r in rows])) if rows else None,
                mean_cell_TTFT_p95_regret_fraction=float(np.mean([r['TTFT_p95_regret_fraction'] for r in rows])) if rows else None))
    write(ROOT/'analysis/c0_observable_axis_rule_diagnostic_v2.json',dict(time=time.time(),complete_heldout_cells=len(cells),target_cells=80,
        rules_SHA256=digest(ROOT/'configs/c0_output_selector_rules_frozen_v2.json'),rows=records,summary=summary,
        scope=frozen['scope']+' Heldout labels use N3 paired-start means. CI-confirmed and decode-qualified subsets remain separate. '
            'Cell-level regret is not a client-latency aggregate or a replayed serving speedup.'))
    csv_save(ROOT/'figures/c0_observable_axis_rule_predictions_v2.csv',records)
    csv_save(ROOT/'figures/c0_observable_axis_rule_summary_v2.csv',summary)
    print('C0_RULE_HELDOUT_DIAGNOSTIC',len(cells),summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','evaluate'));args=p.parse_args();globals()[args.action]()
