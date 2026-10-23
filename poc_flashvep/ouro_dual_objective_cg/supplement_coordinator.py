"""Wait for the existing quality queue, then fill secondary live budget bins."""
import fcntl
import json
import os
import time
from .common import ROOT,write
from .followup import command

def main():
    own=(ROOT/'state'/'budget_supplement.lock').open('a')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lease=(ROOT/'state'/'coordinator.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX)
    plan=json.loads((ROOT/'p4_quality_plan.json').read_text())
    assert all((ROOT/'raw'/j['id']/'completed.json').exists() for j in plan['jobs']), 'Preserve and resolve the main quality queue first'
    write(ROOT/'state'/'supplement.json',dict(pid=os.getpid(),stage='SECONDARY_BUDGET_COVERAGE',wall_s=time.time()))
    write(ROOT/'research_state.json',dict(status='RUNNING',phase='SECONDARY_BUDGET_CALIBRATION',
        next='Complete six diagnostic budget jobs. Primary120ms verdict cannot be retuned from these CALIBRATION outcomes. Then finish final reporting.',
        native_vllm_loop_PP='PAUSED'))
    command('dispatch','--manifest',str(ROOT/'p2_budget_supplement_plan.json'),'--manage-burners')
    for module in ('analyze_serving','report_tables','mechanism_analysis','same_output_diagnostic','analyze_resources','slo_sensitivity','budget_supplement_analyze'):
        command(module)
    write(ROOT/'state'/'supplement.json',dict(pid=os.getpid(),stage='ALL_MEASUREMENTS_COMPLETE_REVIEW_REQUIRED',wall_s=time.time()))
    write(ROOT/'research_state.json',dict(status='MEASUREMENTS_COMPLETE_REVIEW_REQUIRED',phase='FINAL_REPORT_AND_SCOPED_GIT',
        next='Review quality and secondary budget coverage. Validate hashes, export portable evidence, archive raw attempts, restore owned4-7 burners, scoped commit/push.',
        native_vllm_loop_PP='PAUSED'))

if __name__=='__main__':main()
