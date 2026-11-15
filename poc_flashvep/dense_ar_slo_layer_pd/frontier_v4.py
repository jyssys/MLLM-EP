"""Test lower-load Decode frontiers instead of selecting failing goodput points."""
from .common import *
from .configuration import base
from .frontier import run


def main():
    common=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,
        phase='SLO_SPECIFIC_LOWER_LOAD_CONTINUOUS_COLOCATED_DISCOVERY')
    jobs=[]
    for cap,rate,budget,slots in ((16,2.5,256,16),(16,2.5,384,16),
        (16,3.,448,16),(16,3.,512,4),(16,2.5,8192,1),
        (64,2.,256,16),(64,2.,512,16),(64,1.,8192,1),
        (200,.5,8192,1),(200,1.,256,16)):
        jobs.append(base(f'frontier4_C_B{budget}_S{slots}_O{cap}_q{rate}_N128_v2',
            'Colocated',count=128,cap=cap,rate=rate,chunk=budget,token_budget=budget,
            D_slots=slots,**common))
    run(jobs)
    write(ROOT/'state/frontier_v4_complete.json',dict(time=time.time(),
        scope='Discovery pilots only. A high finite goodput at less than95% joint attainment is not a sustainable capacity point.'))


if __name__=='__main__':main()
