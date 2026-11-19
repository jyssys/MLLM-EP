"""Change one scheduler limit at a time on the same pilot workload."""
import json
import sys
import time
from .common import ROOT,write
from .campaign import serving


def main():
    path=ROOT/'configs/frozen_LongBench200_Main_grid_v1.json'
    assert (ROOT/'state/capacity_pilot_complete.json').exists()
    frozen=json.loads(path.read_text());grid=frozen['grid'];rates=[grid[0],grid[2],grid[-1]]
    design=ROOT/'configs/frozen_B_S_sensitivity_design_v1.json'
    values=dict(time=time.time(),offered_QPS=rates,settings=[[8192,1024],[4096,1024],[16384,1024],[8192,256]],
        n=40,dataset='capacity_pilot40.json',arrival_seed=2026112031,independent_starts=1,
        selection='Low, knee estimate, overload candidates from pre-Main pilot. One scheduler variable changes at a time.',
        scope='Discovery sensitivity only. Not a claim of globally best-tuned settings or independent held-out performance.')
    if design.exists():values=json.loads(design.read_text())
    else:write(design,values)
    results=[]
    active_rates=values['offered_QPS'][1:2] if '--moderate-only' in sys.argv else values['offered_QPS']
    for rate in active_rates:
        for b,s in values['settings']:
            for method in ('Native','Agg'):
                name=f'sensitivity_{method}_N40_q{rate:g}_B{b}_S{s}_BI1_v1'
                try:
                    row=serving(name,method,values['dataset'],40,rate,values['arrival_seed'],
                        token_budget=b,max_num_seqs=s,phase='ONE_FACTOR_B_S_SENSITIVITY')
                    results.append(dict(name=name,valid=True,metrics=row))
                except Exception as error:
                    results.append(dict(name=name,valid=False,error=repr(error)))
                    assert not (ROOT/'state/STOP_REQUESTED.json').exists(),'Respect a signal or resource watchdog'
                write(ROOT/'analysis/B_S_sensitivity_v1.json',dict(results=results))
    primary=[r for r in results if '_B8192_S1024_' in r['name']]
    passed=len(primary)==2*len(active_rates) and all(r['valid'] for r in primary)
    write(ROOT/'analysis/B_S_pre_main_gate.json',dict(passed=passed,primary_validated=primary,
        total_attempted=len(results),valid=sum(r['valid'] for r in results),
        evidence='MEASURED_DISCOVERY_SENSITIVITY',
        loads_evaluated=active_rates,scope='Both primary method controls must pass at each evaluated load. The full sensitivity design also covers low and high load. Invalid alternatives remain excluded. This gate does not certify global optimality.'))


if __name__=='__main__':main()
