"""Keep the failed observer startup and resume only unmeasured controls."""
import subprocess
from .common import *


def main():
    initialize();plan_path=ROOT/'state/serial_cost_plan.json';plan=json.loads(plan_path.read_text())
    assert plan['revision']==2
    progress=json.loads((ROOT/'state/runtime_progress.json').read_text());assert progress['closed']
    failed=[r for r in progress['results'] if r['returncode']]
    assert [r['name'] for r in failed]==['cost_ouro_zero_original_serial8_s1']
    folder=ROOT/'raw'/failed[0]['name']
    assert not (folder/'measurement_begin.json').exists(),'Do not relabel a measured failure as startup-only'
    write(ROOT/'state/serial_before_observer_fix.json',progress)
    write(ROOT/'state/serial_cost_plan_v2.json',plan)
    new_jobs=[]
    for name in plan['jobs']:
        if name.startswith('cost_ouro_'):
            cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text());name=name[:-2]+'s2'
            cfg.update(name=name,serial_output_reference='cost_ouro_zero_original_serial8_s2',
                serial_baseline=f'cost_ouro_zero_{"tuned" if cfg["projection_kernel_policy"]=="tuned_s1" else "original"}_serial8_s2',
                workload=f'L{cfg["length"]}_O{cfg["cap"]}',
                data_scope=dict(cfg['data_scope'],distinct_prompt_ids=cfg['count']),
                observer_startup_fix='Ouro has no prepared Prelude/Coda methods. Skip those inapplicable observers. Model computation remains unchanged.')
            save(cfg)
        new_jobs.append(name)
    plan['jobs']=new_jobs;plan['revision']=3;write(plan_path,plan)
    passed={r['name']:r for r in progress['results'] if r['returncode']==0}
    remaining=[name for name in plan['jobs'] if name not in passed]
    code=subprocess.run(['/home/esjung/.venvs/looped-pd-abc/bin/python','-u','-m',
        'poc_flashvep.loop_width_gemm_validation.sequence',*remaining],cwd=REPO).returncode
    result=json.loads((ROOT/'state/runtime_progress.json').read_text())
    assert code==0 and result['closed'] and [r['name'] for r in result['results']]==remaining
    assert all(r['returncode']==0 for r in result['results'])
    passed.update({r['name']:r for r in result['results']})
    names=plan['qualification']+plan['jobs'];assert set(names)<=passed.keys()
    write(ROOT/'state/serial_runtime_completed.json',dict(closed=True,results=[passed[n] for n in names],
        retained_failure=failed,scope='Combined successful immutable cells. The failed original Ouro startup remains separately recorded. No measured failure was discarded.'))
    from .after_costs import main as continue_campaign
    continue_campaign()


if __name__=='__main__':main()
