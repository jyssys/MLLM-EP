"""Run bounded observed-state pilots before freezing confirmation choices."""
import fcntl
from .common import *
from .configuration import base
from .run import execute
from .analyze import compare,metrics,records


def jobs():
    result=[]
    for count,cap in ((8,16),(32,64)):
        result.append(base(f'actual_Revised_O{cap}_N{count}_qualification_v1',arm='Revised',count=count,cap=cap,
            verify_layer_execution=True,phase='ACTUAL_OBSERVED_STATE_SELECTION_QUALIFICATION'))
    for cap in (64,16):
        for arm in ('Native','Revised','STATIC','COST'):
            changes=dict(static_layers=12,migration_fraction=.3125) if arm=='STATIC' else {}
            result.append(base(f'pilot_{arm}_O{cap}_N64_q4p8_s21_v1',arm=arm,cap=cap,**changes))
    for c in result:save(c)
    return result


def main():
    initialize();assert (ROOT/'state/continuous_N32_qualified.json').exists();results=[]
    try:
        for cfg in jobs():
            name=cfg['name']
            if not execute(name,restore=False):
                write(ROOT/'state/pilot_progress.json',dict(results=results,failed=name,reason='RUNTIME_FAILURE'));return
            output=compare('baseline_Native_O200_N64_q4p8_s21_v1',name,True)
            commits=records(ROOT/'raw'/name/'actual_cut_commits.jsonl')
            counts={str(n):sum(len(row['requests']) for row in commits if row['offloaded_layers']==n) for n in (0,2,4,8,12)}
            result=dict(name=name,output=output,completed_layer_distribution=counts,metrics=metrics(name))
            if cfg.get('verify_layer_execution'):
                from ..dense_ar_dynamic_layer_pd import qualify
                qualify.ROOT=ROOT;qualify.state_gate(name,cfg['count'],mixed=True)
                audit=json.loads((ROOT/'analysis'/(name+'_gate.json')).read_text())
                # A correct Native fallback is not a work/lifetime failure.
                # Keep adaptation as a separate measured requirement.
                execution_checks=[r for r in audit['checks'] if r['check']!='MULTIPLE_ACTUAL_CUTS']
                result['work_lifetime_state_pass']=all(r['passed'] for r in execution_checks)
                actual={r['offloaded_layers'] for r in commits}
                result['actual_state_adaptation_demonstrated']=len(actual)>1 and any(actual)
                write(ROOT/'analysis'/(name+'_execution_only_gate.json'),dict(checks=execution_checks,
                    passed=result['work_lifetime_state_pass'],actual_cuts=sorted(actual),
                    actual_adaptation_demonstrated=result['actual_state_adaptation_demonstrated'],
                    scope='Native-only execution can pass correctness but cannot demonstrate migration or state adaptation.'))
            results.append(result);write(ROOT/'state/pilot_progress.json',dict(results=results))
            print('PILOT',name,output['passed'],counts,result['metrics']['TTFT_p95_ms'],result['metrics']['ITL_p99_ms'],flush=True)
            if not output['passed'] or result.get('work_lifetime_state_pass') is False:return
        write(ROOT/'analysis/initial_pilot_results.json',dict(results=results,
            scope='Discovery point estimates. No independent-start non-inferiority or sustainable-capacity claim.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
