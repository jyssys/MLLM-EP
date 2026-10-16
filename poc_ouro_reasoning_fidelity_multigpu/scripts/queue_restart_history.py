"""Run only the predeclared bounded history controls after reference completion."""
import subprocess
import time
import argparse
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def run(name,flag):
    code=subprocess.call([CPU,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.gpu_session',
        '--gpus','4,5,6,7','--output','results/'+name,'--timeout','1200','--',
        PYTHON,'-m','poc_ouro_reasoning_fidelity_multigpu.scripts.run_restart_history',
        '--core-multiprocessing',flag],cwd=PROJECT)
    if code:
        raise RuntimeError('Preserve the failed history probe. Do not reinterpret it as a reproducibility PASS.')
    return read(ROOT/'results'/name/'restart_history_comparison.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--default-output-name',default='restart_same_history_default_v1')
    options=parser.parse_args()
    assert options.default_output_name.startswith('restart_same_history_default_')
    assert '/' not in options.default_output_name
    marker=ROOT/'results/hf_prompt_corroboration_v1/guard_status.json'
    deadline=time.monotonic()+36000
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError('The reference job did not release the assigned GPUs')
        time.sleep(10)
    assert read(marker)['success']
    control=run(options.default_output_name,'1')
    if control['all_equal']:
        write(ROOT/'results/restart_history_gate.json',dict(control=control,additional_MP0_runs='NOT_RUN',
            reason='The fixed-history control matched. Preserve the earlier different-history failure without calling it a universal restart failure.'))
        return
    a=run('restart_same_history_MP0_v1','0')
    b=run('restart_same_history_MP0_v2','0')
    x=read(ROOT/'results/restart_same_history_MP0_v1/request_history.json')[2]['outputs']
    y=read(ROOT/'results/restart_same_history_MP0_v2/request_history.json')[2]['outputs']
    matched=[dict(id=p['id'],equal=p['generated_ids']==q['generated_ids']) for p,q in zip(x,y)]
    write(ROOT/'results/restart_history_gate.json',dict(control=control,MP0_same_history_comparison=matched,
        MP0_pair_equal=all(r['equal'] for r in matched),quality_outputs_replaced=False,
        scope='Generic reproducibility diagnostic only. Do not infer full quality or recurrence-specific novelty.'))


if __name__=='__main__':
    main()
