"""Test the historical cyclic domain order without selecting a favorable order."""
import datetime
import random
from poc_depth_cdb_factorial.contract import *

def main():
    path=ROOT/'artifacts/sensitivity_shuffled_DP4_plan.json'
    assert not path.exists()
    speed=read(ROOT/'artifacts/speed_plan.json')['cases']
    work=read(ROOT/'artifacts/speed_reference_depth_work.json')
    cases=[]
    for C in [32,64]:
        ids=list(next(c['ids'] for c in speed if c['C']==C))
        random.Random(20261019).shuffle(ids)
        for repeat in range(3):
            for arm in (['A1','A2','A3','A4'] if repeat%2==0 else ['A4','A3','A2','A1']):
                placement=lpt(ids,work) if ARMS[arm]['hindsight'] else {rid:i%4 for i,rid in enumerate(ids)}
                cases.append(dict(name=f'order_control_C{C}_{arm}_r{repeat}',kind='order_control',arm=arm,
                    C=C,repeat=repeat,ids=ids,cap=256,placements=placement))
    write(path,dict(cases=cases,local_batch_cap=16,seed=20261019,
        frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Historical request order cycles domains. Round-robin therefore segregates domains by rank. Test one fixed shuffled order, not the best of many orders.',
        original_primary_unchanged=True,normalization='Matched DP4 only. No A0-normalized claim for this changed order.'))
    print('ORDER_CONTROL_FROZEN',len(cases))

if __name__=='__main__':main()
