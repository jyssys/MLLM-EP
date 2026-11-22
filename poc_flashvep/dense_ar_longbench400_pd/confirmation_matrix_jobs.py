"""Freeze all64 held-out cells at one common QPS per input bucket."""
import json
from .common import ROOT,write
from .freeze_capsule import freeze
from .experiment import create
from .geometry import matched_suffixes
from .next_jobs import plan


def main():
    capsule,sha=freeze('c0_heldout_confirmation_r2_20261121')
    seeds=[2026112251,2026112252,2026112253];lanes=[[],[],[],[]];cells=[]
    for bucket in range(4):
        protocol=json.loads((ROOT/f'configs/matrix_input{bucket}_protocol_frozen_v1.json').read_text())
        rates=[r['reference_rate'] for r in protocol['cells']]
        qps=round(.85*min(rates),6)
        dataset=f'input{bucket}_heldout_exact16.json';data=json.loads((ROOT/'data'/dataset).read_text())
        # Keep each matched start on one pair. Input buckets use the same pairs as their discovery.
        first=[2,6,4,0][bucket];lane=first//2
        for output in (8,16,64,128):
            for seed in seeds:
                common=dict(pair=(first,first+1),qps=qps,seed=seed,output=output,fixed=True,controlled=True,
                    qualification_gate='C0_full_state_q4_gate.json',source_capsule=capsule,source_capsule_SHA256=sha,
                    confirmation=True,discovery_only=False,mechanism_input_bucket=bucket,
                    mechanism_output_tokens=output,common_QPS_across_output_lengths=True,
                    primary_capacity_reference=dict(reference='Minimum discovery Native service-demand rate across four outputs',
                        rate=min(rates),multiplier=.85,scope='QPS is frozen from Native pilots only. This is not an observed sustainable capacity.'),
                    request_holdout_scope='Held out from partition discovery. Some prompts appeared in baseline capacity pilots. Contexts do not overlap partition discovery.')
                native=create(f'ConfirmC0_I{bucket}_O{output}_s{seed}_Native_r2','Native',dataset,**common)
                lanes[lane].append(native)
                for cut in (2,4,8,12):
                    suffixes,audit=matched_suffixes([r['input_tokens'] for r in data],cut);assert audit['exact_match']
                    order=('Layer','Token') if (seed+cut//2+output//8)%2 else ('Token','Layer')
                    names={}
                    for axis in order:
                        extra=dict(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},aggregate_work_audit=audit) if axis=='Token' else {}
                        name=create(f'ConfirmC0_I{bucket}_O{output}_C{cut}_s{seed}_{axis}_r2',axis,dataset,cut=cut,**common,**extra)
                        lanes[lane].append(name);names[axis]=name
                    cells.append(dict(input_bucket=bucket,output=output,cut=cut,seed=seed,QPS=qps,
                        Native=native,**names,n_per_start=16))
    manifest=ROOT/'configs/heldout64_confirmation_protocol_r2.json'
    value=dict(cells=cells,seeds=seeds,source_capsule=capsule,source_capsule_SHA256=sha,
        paired_server_starts_per_cell=3,maximum_uncertain_representative_starts=5,
        scope='All64 input/output/cut cells are retained. QPS is constant across output lengths within each input bucket. '
            'This isolates the output contract at the same arrival trace. It complements the capacity-relative discovery grid. '
            'Three independent engine starts,not16 requests,define each performance CI. Native is shared across cuts and counted once. '
            'Layer/Token use exact aggregate GEMM rows,credit4 and actual1/1/1. '
            'Both Layer candidates and Token candidates remain in the confirmation. No SLO or Main400 selection changes.')
    if manifest.exists():assert json.loads(manifest.read_text())==value
    else:write(manifest,value)
    for lane,jobs in enumerate(lanes):plan(f'heldout64_confirmation_lane{lane}_r2',jobs)
    print('Frozen432 server starts for64 held-out cells and their Native controls.',flush=True)


if __name__=='__main__':main()
