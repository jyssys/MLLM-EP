"""Complete missing batch coverage, qualify it, and run bounded discovery."""
import os,subprocess
from .common import *

PYTHON='/home/esjung/.venvs/looped-pd-abc/bin/python'


def command(module,*args):
    assert time.time()<LONG_START_CUTOFF,'No long job can start after the campaign cutoff'
    result=subprocess.run([PYTHON,'-u','-m','poc_flashvep.loop_width_gemm_validation.'+module,*args],cwd=REPO)
    assert result.returncode==0,(module,result.returncode)


def completed(names):
    progress=json.loads((ROOT/'state/runtime_progress.json').read_text())
    assert progress['closed'] and [r['name'] for r in progress['results']]==names
    assert all(r['returncode']==0 for r in progress['results'])
    return progress


def phase(name):
    write(ROOT/'state/post_cost_phase.json',dict(phase=name,time=time.time()))
    print('PHASE',name,flush=True)


def main():
    initialize();previous=json.loads((ROOT/'state/serial_chain_controller.json').read_text())
    while True:
        try:os.kill(previous['pid'],0)
        except ProcessLookupError:break
        assert time.time()<LONG_START_CUTOFF
        time.sleep(5)
    plan=json.loads((ROOT/'state/serial_cost_plan.json').read_text())
    combined=ROOT/'state/serial_runtime_completed.json'
    if combined.exists():
        value=json.loads(combined.read_text())
        assert value['closed'] and [r['name'] for r in value['results']]==plan['qualification']+plan['jobs']
        assert all(r['returncode']==0 for r in value['results'])
    else:write(combined,completed(plan['qualification']+plan['jobs']))
    phase('SERIAL_ANALYSIS');command('cost_analysis')
    serial=json.loads((ROOT/'analysis/serial_path_costs.json').read_text())
    assert len(serial)==len(plan['jobs']) and all(r.get('passed') for r in serial)
    phase('PACKED_CORE_COMPONENT_CONTROL')
    jobs=json.loads((ROOT/'state/core_component_plan.json').read_text())['jobs']
    command('sequence',*jobs)
    write(ROOT/'state/core_component_completed.json',completed(jobs))
    command('cost_analysis')
    serial=json.loads((ROOT/'analysis/serial_path_costs.json').read_text())
    assert len(serial)==len(plan['jobs'])+len(jobs) and all(r.get('passed') for r in serial)
    phase('BATCH_GEMM_COVERAGE')
    write(ROOT/'state/initial_micro_completed.json',json.loads((ROOT/'state/micro_progress.json').read_text()))
    jobs=json.loads((ROOT/'state/batch_coverage_plan.json').read_text())['jobs']
    command('micro_sequence',*jobs)
    micro=json.loads((ROOT/'state/micro_progress.json').read_text())
    assert micro['closed'] and len(micro['results'])==len(jobs) and all(r['returncode']==0 for r in micro['results'])
    write(ROOT/'state/batch_micro_completed.json',micro)
    command('freeze_coverage')
    phase('BATCH_COVERED_N8_N32_QUALIFICATION')
    jobs=json.loads((ROOT/'state/v2_qualification_plan.json').read_text())['jobs']
    command('sequence',*jobs)
    write(ROOT/'state/batch_covered_runtime_qualification_completed.json',completed(jobs))
    phase('MARKER_FREE_N64_DISCOVERY');command('prepare_serving','--version','v2')
    jobs=json.loads((ROOT/'state/serving_v2_discovery_plan.json').read_text())['jobs']
    command('sequence',*jobs)
    write(ROOT/'state/serving_discovery_completed.json',completed(jobs))
    command('serving_analysis')
    results=json.loads((ROOT/'analysis/serving_results.json').read_text())
    assert len([r for r in results if r.get('complete') and r['exact_output_work_lifetime_passed']])==len(jobs)
    phase('DISCOVERY_READY_FOR_REVIEW')


if __name__=='__main__':main()
