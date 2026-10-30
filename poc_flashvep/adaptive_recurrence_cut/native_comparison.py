"""Compare streamed native PD with the explicitly matched snapshot control."""
import json
import statistics
from .common import ROOT,write
from .gate import read
from .analyze import csv_write


def main():
    results=[];checks=[]
    for model,depth in (('ouro',4),('huginn',32)):
        native=ROOT/'raw'/f'map_{model}_L512_native_G1_start3'
        snapshot=ROOT/'raw'/f'map_{model}_L512_frozen_choices_start3'
        if not all((p/'status.json').exists() and json.loads((p/'status.json').read_text())['status']=='COMPLETE' for p in (native,snapshot)):continue
        n=[r for r in read(native/'requests.jsonl') if not r['warmup']]
        s=[r for r in read(snapshot/'requests.jsonl') if not r['warmup'] and r['cohort_id'].endswith(f'_cut{depth}')]
        lookup={(r['regime'],r['seed']):r for r in n}
        for r in s:
            other=lookup.get((r['regime'],r['seed']))
            checks.append(dict(model=model,regime=r['regime'],seed=r['seed'],
                               exact=bool(other and all(r[k]==other[k] for k in ('input_ids','output_ids','exit_depths','finish_reason')))))
        for regime in sorted({r['regime'] for r in n}):
            a=[r for r in n if r['regime']==regime];b=[r for r in s if r['regime']==regime]
            if len(a)!=len(b):raise RuntimeError('Unmatched native/snapshot request denominators')
            row=dict(model=model,regime=regime,N=len(a),independent_paired_starts=1,
                     native_mean_E2E_ms=statistics.mean(r['e2e_ms'] for r in a),
                     snapshot_mean_E2E_ms=statistics.mean(r['e2e_ms'] for r in b),
                     evidence='MEASURED_SINGLE_START_CONTROL',
                     CI95='NOT_ESTIMABLE_N_LT3',
                     limitation='Same physical pair and frozen inputs/seeds. Different server starts. Not sustainable serving capacity.')
            row['snapshot_speedup_vs_native']=row['native_mean_E2E_ms']/row['snapshot_mean_E2E_ms']
            results.append(row)
    csv_write('native_pd_g1_vs_snapshot.csv',results);csv_write('native_pd_g1_output_parity.csv',checks)
    write(ROOT/'state/native_control_parity.json',dict(N=len(checks),exact=sum(r['exact'] for r in checks),
          passed=bool(checks) and all(r['exact'] for r in checks)))
    print('NATIVE_CONTROL_PARITY',sum(r['exact'] for r in checks),'/',len(checks))


if __name__=='__main__':main()
