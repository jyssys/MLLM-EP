"""Export frozen predictions. This module never refits or recalibrates."""
import csv
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import ROOT,write
from . import predictor as q
from . import decode_prediction as d
from .surface_analysis import csv_out

def rows(path):
    with gzip.open(path,'rt') as stream:
        for raw in csv.DictReader(stream):
            row={}
            for k,v in raw.items():
                if v.startswith('[') or v.startswith('{'):
                    row[k]=json.loads(v)
                else:
                    try:
                        row[k]=float(v)
                    except ValueError:
                        row[k]=v
            yield row

def main():
    report=json.loads((ROOT/'predictor_blind_results.json').read_text())
    coverage=[];overhead=[]
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    sources=json.loads((ROOT/'predictor_dataset_sources.json').read_text())['files']
    hashes={r['path']:r['sha256'] for r in sources}
    for target in ('quantum','next_completed_token'):
        selection=json.loads((ROOT/'predictor_artifact'/f'{target}_selection.json').read_text())
        model=selection['model'];table={tuple(r['key']):r['value'] for r in model['table']}
        key_fn=q.key if target=='quantum' else d.key
        predictions=[];residuals=[]
        for ordinal,row in enumerate(rows(ROOT/f'{target}_heldout_dataset.csv.gz')):
            expected=table.get(key_fn(row),model['mean'])
            source=f"raw/{row['run_id']}/{row['policy']}_C{int(row['configured_C'])}_T{int(row['trial'])}.json"
            predictions.append(dict(dataset_row=ordinal,run_id=row['run_id'],trial=int(row['trial']),policy=row['policy'],
                source=source,source_sha256=hashes[source],target_ms=row['target_ms'],expected_ms=expected,
                safe_ms=expected+model['upper_margin_ms'],OOD=key_fn(row) not in table,
                evidence='BLIND_RETROSPECTIVE_PREDICTION_NOT_ONLINE_SCHEDULING'))
            residuals.append(row['target_ms']-expected)
        csv_out(ROOT/f'{target}_runtime_predictor_holdout.csv.gz',predictions)
        for r in report[target]['strata']:
            coverage.append(dict(target=target,stratum=r['stratum'],key=r['key'],observations=r['observations'],
                MAE_ms=r['MAE_ms'],coverage=r['upper_bound_coverage'],CI95_low=r['coverage_CI95'][0],CI95_high=r['coverage_CI95'][1]))
        for r in report[target]['results']:
            overhead.append(dict(target=target,model=r['model'],split=r['split'],selected=r['selected'],query_overhead_us=r['query_overhead_us']))
        x=np.array([0,.25,.5,1,1.5,2]);error=np.array(residuals)
        axes[0].plot(x,[np.mean(error<=z*model['upper_margin_ms'])*100 for z in x],marker='o',label=target)
    phase=[r for r in coverage if r['target']=='quantum' and r['stratum']=='phase']
    axes[1].barh([r['key'] for r in phase],[100*r['coverage'] for r in phase])
    axes[1].axvline(95,color='red',linestyle='--',label='Registered review target')
    axes[1].set(xlabel='Selected quantum bound coverage (%)',title='Blind phase-specific coverage')
    axes[0].set(xlabel='Multiplier of frozen margin (sensitivity only)',ylabel='Blind bound coverage (%)',title='No sensitivity margin is adopted')
    axes[0].legend();axes[0].grid(alpha=.25);axes[1].legend(fontsize=8)
    fig.savefig(ROOT/'figures/predictor_risk.png',dpi=170);fig.savefig(ROOT/'figures/predictor_risk.svg');plt.close(fig)
    csv_out(ROOT/'prediction_coverage_by_shape.csv',coverage)
    csv_out(ROOT/'predictor_runtime_overhead.csv',overhead)
    write(ROOT/'not_run_tables.json',dict(online_action_decisions='NOT_RUN: Phase3/4 gate blocked',
        slo_attainment_by_rate='NOT_RUN for new adaptive policies',server_backlog_timeseries='Only diagnostic queue_series exists. No qualified capacity lane.',
        false_feasible_selection='NOT_RUN. Bound miss rate is not false-feasible scheduling rate.'))
    print('Frozen prediction exports complete. No fit or threshold change.')

if __name__=='__main__':
    main()
