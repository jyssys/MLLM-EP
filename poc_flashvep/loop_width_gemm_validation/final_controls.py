"""Run realized quantum calibration and the frozen confirmation plan."""
import subprocess
from .common import *


def command(module,*args):
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.loop_width_gemm_validation.'+module,*args],cwd=REPO).returncode
    assert code==0,(module,code)


def main():
    initialize();assert json.loads((ROOT/'state/post_cost_phase.json').read_text())['phase']=='DISCOVERY_READY_FOR_REVIEW'
    plan=json.loads((ROOT/'state/v2_serial_quantum_plan.json').read_text())
    command('sequence',*plan['jobs'])
    value=json.loads((ROOT/'state/runtime_progress.json').read_text())
    assert value['closed'] and [r['name'] for r in value['results']]==plan['jobs'] and all(r['returncode']==0 for r in value['results'])
    write(ROOT/'state/v2_serial_quantum_completed.json',value)
    command('cost_analysis')
    values=json.loads((ROOT/'analysis/serial_path_costs.json').read_text());assert all(r.get('passed') for r in values)
    command('confirm');command('serving_analysis');command('confirmation_stats')
    write(ROOT/'state/final_controls_completed.json',dict(time=time.time(),complete=True))


if __name__=='__main__':main()
