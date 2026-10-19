"""Build immutable execution plans from the frozen factorial design."""
import argparse
import gzip
import hashlib
from poc_depth_cdb_factorial.contract import *

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--quality-followup',type=Path)
    args=p.parse_args()
    if args.quality_followup:
        raw=[]
        paths=sorted({Path(str(p).removesuffix('.gz')) for p in (args.quality_followup/'rank_0').glob('quality_A0_part*.json*')})
        for path in paths:
            raw+=read(path)['outputs']
        frozen=read(ROOT/'artifacts/workload_frozen.json')
        expected={r['id'] for r in frozen['quality']}
        assert {r['id'] for r in raw}==expected
        work={r['id']:sum(e['depth'] for e in r['depths']) for r in raw}
        reference=dict(source=str(args.quality_followup),work=work,
            raw_sha256={str(p):hashlib.sha256(p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())).hexdigest() for p in paths},
            answer_labels_used=False,placement_rule='LPT of sum of actual decode depths, frozen before A2/A4.')
        write(ROOT/'artifacts/quality_hindsight_reference.json',reference)
        template=read(ROOT/'artifacts/quality_A0_plan.json')['cases']
        cases=[]
        for index,base in enumerate(template):
            order=['A1','A2','A3','A4'] if index%2==0 else ['A4','A3','A2','A1']
            for arm in order:
                case=dict(base,name=base['name'].replace('A0',arm),arm=arm)
                case['placements']=lpt(case['ids'],work) if ARMS[arm]['hindsight'] else {rid:i%4 for i,rid in enumerate(case['ids'])}
                cases.append(case)
        write(ROOT/'artifacts/quality_DP4_plan.json',dict(cases=cases,local_batch_cap=16))
        return
    speed=read(ROOT/'artifacts/speed_plan.json')
    for repeat in range(3):
        for dp in [False,True]:
            selected=[c for c in speed['cases'] if c['repeat']==repeat and (c['arm']!='A0')==dp]
            write(ROOT/f'artifacts/speed_{"DP4" if dp else "A0"}_r{repeat}_plan.json',dict(cases=selected,local_batch_cap=16))
    # Trace the same C32/C64 requests. These runs are not headline wall-time samples.
    selected=[dict(c,name='trace_'+c['name']) for c in speed['cases']
        if c['repeat']==0 and c['C'] in [32,64] and c['arm']!='A0']
    write(ROOT/'artifacts/trace_DP4_plan.json',dict(cases=selected,local_batch_cap=16))
    write(ROOT/'artifacts/trace_A0_plan.json',dict(cases=[dict(c,name='trace_'+c['name']) for c in speed['cases']
        if c['repeat']==0 and c['C'] in [32,64] and c['arm']=='A0'],local_batch_cap=16))
    # The historical stronger search is a named sensitivity, not a selected main winner.
    historical=read(V3_ROOT/'artifacts/oracle_refinement_v2.json')
    selected=next(r for r in historical['placements'] if r['C']==64)
    placement={r['id']:r['rank'] for r in selected['placements']}
    extra=[]
    for repeat in range(3):
        for arm in (['A2','A4'] if repeat%2==0 else ['A4','A2']):
            base=next(c for c in speed['cases'] if c['C']==64 and c['arm']==arm and c['repeat']==repeat)
            extra.append(dict(base,name=f'sensitivity_v3_C64_{arm}_r{repeat}',kind='sensitivity',placements=placement))
    write(ROOT/'artifacts/sensitivity_DP4_plan.json',dict(cases=extra,local_batch_cap=16))
    print('PLANS_READY')

if __name__=='__main__':main()
