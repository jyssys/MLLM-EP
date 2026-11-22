"""Use Native service pilots to freeze capacity-relative axis discovery cells."""
import argparse
import json
import numpy as np
from .common import ROOT,write,digest
from .experiment import create
from .next_jobs import plan
from .geometry import matched_suffixes
from .service_accounting import analyze


def main():
    parser=argparse.ArgumentParser();parser.add_argument('bucket',type=int);parser.add_argument('gpu',type=int)
    args=parser.parse_args();bucket=args.bucket;assert 0<=bucket<4 and args.gpu in (0,2,4,6)
    pair=(args.gpu,args.gpu+1);jobs=[];cells=[]
    for output in (8,16,64,128):
        pilot=f'CapacityC0_I{bucket}_O{output}_Serial16_v1'
        values=analyze(pilot);means=values['means']
        bottleneck_ms=max(means['P_submit_to_ready_ms'],means['D_submit_to_finish_ms'])
        reference_rate=1000/bottleneck_ms
        dataset=f'input{bucket}_discovery_exact16.json';rows=json.loads((ROOT/'data'/dataset).read_text())
        lengths=[r['input_tokens'] for r in rows]
        for load_index,load in enumerate((.65,1.15)):
            qps=round(load*reference_rate,6);seed=2026112190+load_index
            group=[]
            base=dict(pair=pair,qps=qps,seed=seed,output=output,fixed=True,controlled=True,
                qualification_gate='C0_full_state_q4_gate.json',
                capacity_reference=dict(pilot=pilot,model='1/max(mean_P_host_service,mean_D_host_cycle)',
                    reference_requests_per_second=reference_rate,relative_load=load,
                    qualifier='Service-demand load estimate. Not a measured sustainable capacity.'),
                mechanism_input_bucket=bucket,mechanism_output_tokens=output,
                mechanism_load_index=load_index,discovery_only=True)
            group.append(create(f'MapC0_I{bucket}_O{output}_U{load_index}_Native_v1','Native',dataset,**base))
            for cut in (2,4,8,12):
                suffixes,audit=matched_suffixes(lengths,cut);assert audit['exact_match']
                order=('Layer','Token') if (bucket+output//8+cut//2+load_index)%2 else ('Token','Layer')
                for axis in order:
                    extra=dict(suffix_by_request={'q'+str(i):s for i,s in enumerate(suffixes)},
                        aggregate_work_audit=audit) if axis=='Token' else {}
                    group.append(create(f'MapC0_I{bucket}_O{output}_U{load_index}_{axis}{cut}_v1',
                        axis,dataset,cut=cut,**base,**extra))
            jobs.extend(group)
            cells.append(dict(input_bucket=bucket,output=output,relative_load=load,qps=qps,
                pilot=pilot,reference_rate=reference_rate,n=16,jobs=group))
    frozen=ROOT/f'configs/matrix_input{bucket}_protocol_frozen_v1.json'
    value=dict(cells=cells,pair=list(pair),evidence='FROZEN_DISCOVERY_DESIGN',
        scope='All four output lengths and cuts are retained. Each cell has one independent start. '
            'No start-level CI or confirmed winner follows from this discovery alone. '
            'Layer and Token share prompts,arrival offsets,credit4,actual1/1/1 and exact aggregate GEMM rows.')
    if frozen.exists():assert json.loads(frozen.read_text())==value
    else:write(frozen,value)
    plan(f'axis_matrix_input{bucket}_discovery_v1',jobs)
    print('Frozen',bucket,'jobs',len(jobs),flush=True)


if __name__=='__main__':main()
