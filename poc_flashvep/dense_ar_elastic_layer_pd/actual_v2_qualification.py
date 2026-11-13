"""Audit real state-aware cuts without forcing their layer counts."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,records


def main():
    cfg=base('actual_Revised_v2_O16_N32_q6_qualification',arm='Revised',count=32,cap=16,rate=6.,
        revised_revision=2,cost_model='revised_cost_model_v2.json',maximum_predicted_D_wait_ms=250,
        maximum_queue_groups=8,D_admission_batch=8,D_token_budget=64000,max_reserved_requests=32,
        defer_completion_log=True,verify_layer_execution=True,hidden_verify_hash=False,
        phase='REAL_STATE_AWARE_N32_WORK_AND_LIFETIME_QUALIFICATION')
    try:
        if not execute(save(cfg),restore=False):return
        output=compare('baseline_Native_O200_N64_q4p8_s21_v1',cfg['name'],True)
        from ..dense_ar_dynamic_layer_pd import qualify
        qualify.ROOT=ROOT;qualify.state_gate(cfg['name'],32,mixed=True)
        gate=json.loads((ROOT/'analysis'/(cfg['name']+'_gate.json')).read_text())
        commits=records(ROOT/'raw'/cfg['name']/'actual_cut_commits.jsonl')
        actual=sorted({r['offloaded_layers'] for r in commits})
        work_checks=[r for r in gate['checks'] if r['check']!='MULTIPLE_ACTUAL_CUTS']
        result=dict(name=cfg['name'],output=output,work_lifetime_pass=all(r['passed'] for r in work_checks),
            actual_layer_counts=actual,actual_state_adaptation=len(actual)>1 and any(actual),
            passed=output['passed'] and gate['passed'] and any(actual),
            scope='Actual observed-state decisions before prefix submission. No forced layer count. Instrumented work audit is not marker-free performance evidence.')
        write(ROOT/'analysis/actual_v2_N32_qualification.json',result);print('ACTUAL_V2_QUALIFICATION',result['passed'],actual,flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
