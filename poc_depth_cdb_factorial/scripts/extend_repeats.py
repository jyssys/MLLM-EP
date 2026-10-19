"""Freeze two extra paired repeats because the initial time estimates vary."""
import datetime
from poc_depth_cdb_factorial.contract import *

def main():
    assert not (ROOT/'artifacts/repeat_extension.json').exists()
    initial=read(ROOT/'artifacts/speed_plan.json')
    cases=[]
    for repeat in [3,4]:
        for C in [4,8,16,32,64]:
            for arm in (['A4','A3','A2','A1','A0'] if repeat%2 else ['A0','A1','A2','A3','A4']):
                base=next(c for c in initial['cases'] if c['C']==C and c['arm']==arm and c['repeat']==0)
                cases.append(dict(base,name=f'speed_C{C}_{arm}_r{repeat}',repeat=repeat))
        for dp in [False,True]:
            selected=[c for c in cases if c['repeat']==repeat and (c['arm']!='A0')==dp]
            write(ROOT/f'artifacts/speed_{"DP4" if dp else "A0"}_r{repeat}_plan.json',dict(cases=selected,local_batch_cap=16))
    write(ROOT/'artifacts/repeat_extension.json',dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        total_repeats=5,added_repeats=[3,4],cases=cases,
        reason='The initial three-repeat C64 absolute-scaling interval is broad. Add two complete paired repeats for every arm and C.',
        selection='Precision extension, not selection of positive cells. No further repeat count tuning is planned.',
        primary_rule_unchanged=True,cohorts_unchanged=True,all_added_cells_required=True))
    print('FIVE_REPEAT_EXTENSION_FROZEN')

if __name__=='__main__':main()
