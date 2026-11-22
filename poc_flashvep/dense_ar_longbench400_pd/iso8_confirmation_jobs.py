"""Freeze a genuine interior hybrid and two exact iso-work pure boundaries."""
import json
import numpy as np
from .common import ROOT,write
from .experiment import create
from .freeze_capsule import freeze
from .next_jobs import plan


def main():
    capsule,sha=freeze('iso8_hybrid_heldout_confirmation_v1_20261121')
    sets=json.loads((ROOT/'data/iso8_hybrid_sets_frozen_v1.json').read_text())['rows']
    cells=[]
    for bucket in range(4):
        jobs=[];first=2*bucket
        data=next(r for r in sets if r['split']=='heldout' and r['input_bucket']==bucket)
        native_protocol=json.loads((ROOT/f'configs/matrix_input{bucket}_protocol_frozen_v1.json').read_text())
        for output in (16,64):
            discovery=next(c for c in native_protocol['cells'] if c['output']==output and c['relative_load']==1.15)
            for start,seed in enumerate((2026112351,2026112352,2026112353)):
                group=[];records={}
                for axis in ('Native','Layer','Token','Hybrid'):
                    extra={}
                    if axis=='Hybrid':extra.update(hybrid_suffix_fraction=.125,partition_aligned_chunks=True,
                        suffix_by_request=data['Hybrid_suffix_by_request'],qualification_gate='C0_family_q7_gate.json')
                    elif axis!='Native':extra['qualification_gate']='C0_full_state_q4_gate.json'
                    if axis=='Token':extra.update(suffix_by_request=data['Token_suffix_by_request'],aggregate_work_audit=dict(
                        target_D_rows=data['D_rows_each'],actual_D_rows=data['D_rows_each'],signed_error_rows=0,
                        absolute_error_fraction=0,exact_match=True,alignment_tokens=16))
                    name=create(f'Iso8_Heldout_I{bucket}_O{output}_{axis}_s{seed}_v1',axis,data['dataset'],
                        cut=4 if axis=='Hybrid' else 8 if axis!='Native' else 0,
                        pair=(first,first+1),qps=discovery['qps'],seed=seed,
                        output=output,fixed=True,controlled=True,source_capsule=capsule,source_capsule_SHA256=sha,
                        iso_work_frontier=True,target_D_rows=data['D_rows_each'],target_P_rows=data['P_rows_each'],
                        scope='Held-out original prompts. Layer8,Token8 and Hybrid4/suffix≈L/8 have exactly equal aggregate token-layer rows. '
                            'Attention work and payloads are not equal. C0 actual1/1/1. This does not establish continuous serving gain.',**extra)
                    group.append(name);records[axis]=name
                np.random.default_rng(seed+bucket*19+output).shuffle(group)
                jobs.extend(group);cells.append(dict(input_bucket=bucket,output=output,QPS=discovery['qps'],
                    seed=seed,start=start,dataset=data['dataset'],D_rows_each=data['D_rows_each'],P_rows_each=data['P_rows_each'],**records))
        plan(f'iso8_heldout_confirmation_lane{bucket}_v1',jobs)
    write(ROOT/'configs/iso8_hybrid_confirmation_frozen_v1.json',dict(cells=cells,source_capsule_SHA256=sha,
        evidence='FROZEN_CONFIRMATION_DESIGN',independent_starts_per_cell=3,
        scope='All eight input/output conditions are retained. Same discovery-defined Native-relative load,not measured sustainable capacity. '
            'These iso-work tests do not replace best-versus-best static tuning. Decode gates and Native-relative gains remain separate.'))
    print('Frozen96 exact-work native/pure/interior confirmation starts.',flush=True)


if __name__=='__main__':main()
