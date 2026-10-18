"""Physically test fixed hindsight placements. This is not a deployable depth predictor."""
import subprocess
import time
import argparse
from poc_decode_first_depth_aware_dp_v3.contract import *

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--concurrency',nargs='+',type=int,default=[16,32])
    parser.add_argument('--tag',default='')
    args=parser.parse_args()
    while not (ROOT/'outputs/batched_kv_probe_v2/guard_status.json').exists():time.sleep(1)
    assert read(ROOT/'outputs/batched_kv_probe_v2/guard_status.json')['success']
    refinement=read(ROOT/'artifacts/oracle_refinement_v2.json')
    base=read(ROOT/'artifacts/live_clean_plan.json')
    metrics=read(ROOT/'artifacts/live_dp4_metrics.json')
    cases=[]
    frozen=[]
    for c in args.concurrency:
        eligible=[r for r in metrics if not r['trace'] and r['arrival']=='burst' and r['concurrency']==c]
        control=min(eligible,key=lambda r:r['E2E_P95_ms'])
        record=read(ROOT/'outputs/live_clean_v1'/(control['case']+'.json'))
        control_placement={r['id']:r['rank'] for r in record['assignments']}
        oracle=next(p for p in refinement['placements'] if p['C']==c)
        oracle_placement={r['id']:r['rank'] for r in oracle['placements']}
        ids=[r['id'] for r in record['assignments']]
        for repeat in range(3):
            # Alternate arm order. The per-request model, gate, and cap are unchanged.
            arms=['generic','hindsight'] if repeat%2==0 else ['hindsight','generic']
            for arm in arms:
                cases.append(dict(name=f'confirmation_C{c}_{arm}_r{repeat}',concurrency=c,
                    arrival='burst',policy='preassigned',ids=ids,
                    placements=control_placement if arm=='generic' else oracle_placement,
                    cap=256,arm=arm,repeat=repeat))
        frozen.append(dict(C=c,control_policy=control['policy'],
            source_refinement_row=next(r for r in refinement['rows'] if r['C']==c),
            scope='Same exploratory held-out pilot. Three physical repeats, not a fresh selection-independent cohort.'))
    suffix='_'+args.tag if args.tag else ''
    plan=ROOT/('artifacts/hindsight_confirmation'+suffix+'_frozen.json')
    assert not plan.exists()
    write(plan,dict(**{k:v for k,v in base.items() if k!='cases'},cases=cases,selection=frozen,
        causal=False,future_information=True,no_RL=True,
        method='Frozen B1-trace clairvoyant placement evaluated with a fresh live online gate. No exit replay.',
        quality_warning='Different batch shapes can change natural outputs. Report parity; do not assume invariant depth traces.'))
    subprocess.run([REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.gpu_session',
        '--gpus','4,5,6,7','--output','outputs/hindsight_confirmation'+suffix+'_v1','--timeout','3600','--',
        REFERENCE_PYTHON,'-m','poc_decode_first_depth_aware_dp_v3.scripts.live_dp','--plan',str(plan)],
        cwd=PROJECT,check=True)

if __name__=='__main__':main()
