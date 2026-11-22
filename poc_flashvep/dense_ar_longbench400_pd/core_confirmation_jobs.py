"""Predeclare five starts for three core cells before any held-out axis result."""
import json
import time
from .common import ROOT,write
from .experiment import create
from .geometry import matched_suffixes
from .next_jobs import plan


def main():
    target=ROOT/'configs/core_five_start_confirmation_frozen_v1.json'
    if not target.exists():
        assert not list((ROOT/'analysis').glob('ConfirmC0*Layer_r2_metrics.json'))
        assert not list((ROOT/'analysis').glob('ConfirmC0*Token_r2_metrics.json'))
    source=json.loads((ROOT/'configs/heldout64_confirmation_protocol_r2.json').read_text())
    chosen=[(0,8,4),(0,8,8),(3,16,8)];lanes={0:[],2:[]};cells=[]
    for bucket,output,cut in chosen:
        reference=next(c for c in source['cells'] if (c['input_bucket'],c['output'],c['cut'])==(bucket,output,cut))
        first={0:2,3:0}[bucket];data=json.loads((ROOT/f'data/input{bucket}_heldout_exact16.json').read_text())
        suffixes,audit=matched_suffixes([r['input_tokens'] for r in data],cut);assert audit['exact_match']
        for seed in (2026112254,2026112255):
            common=dict(pair=(first,first+1),qps=reference['QPS'],seed=seed,output=output,fixed=True,
                controlled=True,qualification_gate='C0_full_state_q4_gate.json',
                source_capsule=source['source_capsule'],source_capsule_SHA256=source['source_capsule_SHA256'],
                confirmation=True,core_five_start=True,mechanism_input_bucket=bucket,
                mechanism_output_tokens=output,common_QPS_across_output_lengths=True,
                scope='Five starts are fixed before any held-out axis metric. This is not outcome-dependent optional stopping.')
            group=[create(f'ConfirmC0_I{bucket}_O{output}_s{seed}_Native_r2','Native',
                f'input{bucket}_heldout_exact16.json',**common)]
            arms={}
            for axis in (('Layer','Token') if seed%2 else ('Token','Layer')):
                extra=dict(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},aggregate_work_audit=audit) if axis=='Token' else {}
                name=create(f'ConfirmC0_I{bucket}_O{output}_C{cut}_s{seed}_{axis}_r2',axis,
                    f'input{bucket}_heldout_exact16.json',cut=cut,**common,**extra)
                group.append(name);arms[axis]=name
            lanes[first].extend(n for n in group if n not in lanes[first])
            cells.append(dict(input_bucket=bucket,output=output,cut=cut,seed=seed,QPS=reference['QPS'],Native=group[0],**arms))
    value=dict(chosen_cells=chosen,additional_cells=cells,seeds=source['seeds']+[2026112254,2026112255],
        rationale='Short-input cut4 and cut8 have opposite one-start discovery signs. '
            'Long-input cut8 is a predeclared Token-favorable structural counterexample. '
            'The low common-QPS confirmation tests intrinsic online paths,not prefill saturation. '
            'The three cells are retained regardless of sign. All other64-grid cells retain their three starts.',
        chosen_before_heldout_axis_results=True,source_capsule_SHA256=source['source_capsule_SHA256'])
    if target.exists():assert json.loads(target.read_text())==value
    else:write(target,value)
    for first,jobs in lanes.items():plan(f'core_confirmation_pair{first}_extra_two_v1',jobs)
    print('Frozen16 additional starts. Three core cells use exactly five starts.',flush=True)


if __name__=='__main__':main()
