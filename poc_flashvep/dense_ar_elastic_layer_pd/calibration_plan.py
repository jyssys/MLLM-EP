"""Prepare legal independent capacities and fresh primitive cost checks."""
from .common import *
from .configuration import base


def main():
    jobs=[]
    p=base('capacity_P_normfix_B124_fresh_v1',count=16,cap=16,dataset='fresh_calibration.json')
    p.update(module='capacity',capacity_mode='P_ONLY',gpus=[0],hidden_transport=None,
        jobs=[dict(batch=b,repeats=3) for b in (1,2,4)],phase='INDEPENDENT_P_CAPACITY_NOT_ONLINE_SERVING')
    jobs.append(p)
    p=base('capacity_D_normfix_B14816_O1664200_fresh_v1',count=16,cap=200,dataset='fresh_calibration.json')
    p.update(module='capacity',capacity_mode='PREPARED_D',gpus=[1],hidden_transport=None,
        jobs=[dict(batch=b,cap=cap,repeats=2) for b in (1,4,8,16) for cap in (16,64,200)],
        phase='VALID_RESIDENT_KV_DECODE_CAPACITY_NOT_ONLINE_SERVING',wall_timeout=1200)
    jobs.append(p)
    # Two repetitions per legal shape are a consistency check, not a p99 bound.
    for batch in (1,4):
        template=f'cost_mixed_B{batch}_N{batch*20}_O2_v1'
        c=json.loads((PREVIOUS/'configs'/(template+'.json')).read_text())
        count=batch*10;c.update(name=f'fresh_cost_B{batch}_Last024812_N{count}_O2_v1',
            dataset='fresh_calibration.json',count=count,request_caps=[2]*count,arrival_offsets=[0.]*count,
            phase_stable_norm=True,phase='FRESH_EVENT_ENCLOSURE_COST_CHECK_NOT_PRIMARY_SERVING')
        # B4 requires40 distinct sources. Use32 fresh calibration documents plus
        #8 old calibration documents. Neither set belongs to fresh confirmation.
        if count>32:
            pool=json.loads((ROOT/'data/fresh_calibration.json').read_text())+json.loads((ROOT/'data/calibration.json').read_text())[:8]
            write(ROOT/'data/fresh_cost40.json',pool);c['dataset']='fresh_cost40.json'
        jobs.append(c)
    for c in jobs:save(c)
    write(ROOT/'state/calibration_plan.json',dict(jobs=[c['name'] for c in jobs],
        scope='Separate capacity and primitive diagnostics. No synthetic fillers or fresh confirmation prompts.'))
    print(' '.join(c['name'] for c in jobs))


if __name__=='__main__':main()
