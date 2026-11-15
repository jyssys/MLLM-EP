"""Retune all three objectives after the standard async option qualifies."""
from .common import *
from .configuration import base
from .frontier import run


def main():
    assert json.loads((ROOT/'analysis/asynchronous_colocated_qualification.json').read_text())['passed']
    jobs=[]
    for budget,cap,rate,slots in ((128,16,3.5,16),(256,16,3.5,16),(384,16,3.5,16),
        (448,16,3.5,16),(512,16,3.5,16),(640,16,3.5,16),
        (8192,16,6.,32),(32768,16,6.,32),
        (256,64,2.5,16),(384,64,2.5,16),(512,64,2.5,16)):
        name=f'frontier_async_C_B{budget}_S{slots}_O{cap}_q{rate}_N128_v1'
        jobs.append(base(name,'Colocated',count=128,cap=cap,rate=rate,chunk=budget,
            token_budget=budget,D_slots=slots,verified_ByteLevel_priming=True,
            direct_phase_norm=True,ar_async_scheduling=True,
            phase='ASYNC_STANDARD_CONTINUOUS_SLO_SPECIFIC_FRONTIER_DISCOVERY'))
    run(jobs)
    write(ROOT/'state/async_frontier_complete.json',dict(time=time.time(),
        scope='Do not select a configuration from a different SLO objective. Peak fixed-window tests follow discovery.'))


if __name__=='__main__':main()
