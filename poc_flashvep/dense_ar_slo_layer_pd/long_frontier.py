"""Check long-output Decode baselines independently of the short-output winner."""
from .common import *
from .configuration import base
from .frontier import run


def main():
    assert json.loads((ROOT/'analysis/asynchronous_colocated_qualification.json').read_text())['passed']
    jobs=[]
    for budget,slots,rate in ((8192,1,.3),(128,8,1.),(256,8,1.),(384,8,1.),(512,8,1.)):
        name=f'long_frontier_C_B{budget}_S{slots}_O200_q{rate}_N128_v1'
        jobs.append(base(name,'Colocated',count=128,cap=200,rate=rate,
            chunk=budget,token_budget=budget,D_slots=slots,ar_async_scheduling=True,
            verified_ByteLevel_priming=True,direct_phase_norm=True,
            phase='LONG_OUTPUT_SLO_SPECIFIC_COLOCATED_DISCOVERY'))
    run(jobs)
    write(ROOT/'state/long_frontier_complete.json',dict(time=time.time(),
        scope='Full200-token greedy parity is checked. These finite pilots are not sustainable capacity.'))


if __name__=='__main__':main()
