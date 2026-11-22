"""Freeze a full static2D discovery and separate serialized cost calibration."""
import copy
import json
import numpy as np
from .common import ROOT,write,digest
from .experiment import create
from .freeze_capsule import freeze
from .next_jobs import plan
from .geometry import Partition


def main():
    capsule,sha=freeze('hybrid_static_discovery_v1_20261121')
    family=json.loads((ROOT/'configs/family_full_state_qualification_frozen_v1.json').read_text())
    actions=[a for a in family['actions'] if a['kind']=='Hybrid']
    cells=[]
    for bucket,first in enumerate((2,6,4,0)):
        jobs=[];protocol=json.loads((ROOT/f'configs/matrix_input{bucket}_protocol_frozen_v1.json').read_text())
        data=json.loads((ROOT/f'data/input{bucket}_discovery_exact16.json').read_text())
        for output in (16,64):
            prior=next(c for c in protocol['cells'] if c['output']==output and c['relative_load']==1.15)
            base=dict(pair=(first,first+1),qps=prior['qps'],seed=2026112191,output=output,
                fixed=True,controlled=True,qualification_gate='C0_family_q7_gate.json',
                source_capsule=capsule,source_capsule_SHA256=sha,
                hybrid_discovery=True,mechanism_input_bucket=bucket,
                capacity_reference=prior,search_contract='All24 non-native2D grid actions are retained. '
                    'This is best-versus-best tuning,not an iso-work comparison. '
                    'Pure boundaries and interior partitions use the same actual1/1/1 contract. '
                    'The independent iso8 comparison controls aggregate work separately.')
            native=create(f'HybridGrid_I{bucket}_O{output}_Native_v1','Native',
                f'input{bucket}_discovery_exact16.json',**base)
            group=[native]
            order=list(actions);np.random.default_rng(2026112280+bucket*100+output).shuffle(order)
            rows=[]
            for action in order:
                m,f=action['cut'],action['suffix_fraction'];suffixes=[];work=[]
                for row in data:
                    length=row['input_tokens'];s=0 if not f else min((length-1)//16*16,max(16,int(round(length*f/16))*16))
                    suffixes.append(s);work.append(Partition(length,m,s).record())
                name=create(f'HybridGrid_I{bucket}_O{output}_m{m}_f{f}_v1','Hybrid',
                    f'input{bucket}_discovery_exact16.json',cut=m,hybrid_suffix_fraction=f,
                    partition_aligned_chunks=True,suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},
                    actual_partition_work=work,**base)
                group.append(name);rows.append(dict(name=name,m=m,fraction=f,
                    boundary='TOKEN_ONLY' if m==0 else 'LAYER_ONLY' if not f else 'INTERIOR_HYBRID',
                    P_rows=sum(r['P_rows'] for r in work),D_rows=sum(r['D_rows'] for r in work)))
            jobs.extend(group);cells.append(dict(input_bucket=bucket,output=output,QPS=prior['qps'],
                Native=native,actions=rows))
        plan(f'hybrid_static_grid_input{bucket}_v1',jobs)
    write(ROOT/'configs/hybrid_static_search_frozen_v1.json',dict(cells=cells,source_capsule=capsule,
        source_capsule_SHA256=sha,actions=actions,server_starts=200,
        scope='One-start discovery. Freeze static winners before held-out tests. '
            'Retain every unfavorable,failed or unmeasured grid action. No performance CI follows from discovery.'))
    original=json.loads((ROOT/'data/family_qualification_actions208.json').read_text())
    mapping=family['mapping']
    for start,first in enumerate((2,4,6)):
        order=np.random.default_rng(2026112287+start).permutation(len(original));data=[];amap={};suffixes={}
        for index,old in enumerate(order):
            data.append(copy.deepcopy(original[int(old)]));amap['q'+str(index)]=mapping[int(old)]['action'];suffixes['q'+str(index)]=mapping[int(old)]['suffix']
        path=ROOT/f'data/cost_calibration_family208_start{start}_v1.json'
        if path.exists():assert json.loads(path.read_text())==data
        else:write(path,data)
        name=create(f'HybridFamily_CostLUT_Serial208_Start{start}_v1','Hybrid',path.name,cut=12,
            pair=(first,first+1),output=16,fixed=True,controlled=True,serial=True,
            hybrid_suffix_fraction=.125,partition_aligned_chunks=True,
            axis_actions_by_request=amap,suffix_by_request=suffixes,
            qualification_gate='C0_family_q7_gate.json',source_capsule=capsule,source_capsule_SHA256=sha,
            wall_timeout=1800,serving_timeout=1500,
            diagnostic_scope='Serialized host-service lookup calibration. Eight original prompts repeated across26 actions. '
                'This is not online throughput or GPU-exclusive kernel time.')
        plan(f'hybrid_cost_LUT_start{start}_v1',[name])
    print('Frozen200 static grid starts and three separate serialized calibration starts.',flush=True)


if __name__=='__main__':main()
