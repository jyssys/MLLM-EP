"""Test whether decode-process profiling caused the failed observer gate."""
import time
from .common import ROOT,write
from .campaign import serving
from .serving_analysis import analyze
from .online_profile import summarize_triplet


def main():
    declaration=ROOT/'configs/frozen_P_only_activity_intervention_v2.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),N=200,QPS=2.35,arrival_seed=2026112051,
            families=['ContinuousNative','ControlledNative'],profile_roles=['prefill'],KV_block=128,
            restore_official_UCX_init=True,observer_limit=.05,clock_bracket_limit_ms=.5,
            hypothesis='Recording decode kernels perturbs short BI0 steps. Removing only the D collector may reduce observer effects.',
            falsifier='A remaining >5% change in any frozen latency statistic rejects primary kernel interpretation.',
            scope='Independent profiling intervention after the two-role observer failed. No relaxed observer threshold. '
            'Only P kernel activity is measured. D host quiescence is not a physical idle timeline or future executable budget.'))
    results=[]
    for family,controlled in (('ContinuousNative',False),('ControlledNative',True)):
        names={}
        for label in ('clean_before','profile','clean_after'):
            name=f'POnly_{family}_N200_q2.35_{label}_BI0_v2';names[label]=name
            extra=dict(controlled_pipeline=True,source_credit_limit=4,execution_family='MATCHED_NO_COMPILE',
                       async_scheduling=False) if controlled else {}
            serving(name,'Native','longbench_shortanswer200_v2.json',200,2.35,2026112051,
                restore_official_UCX_init=True,KV_block_tokens=128,phase='P_ONLY_OBSERVER_INTERVENTION',
                kernel_activity=label=='profile',kernel_activity_roles=['prefill'],
                kernel_activity_seconds=90,quiescent_witness=True,**extra)
            analyze(name)
        result=summarize_triplet(2.35,names,roles=('prefill',));result['measured_roles']=['prefill']
        result['scope']='P-only kernel recording with frozen observer tests. D state is not D kernel activity. No active-decode window or sustainable-capacity claim.'
        write(ROOT/'analysis'/f'BI0_P_only_{family}_observer_v2.json',result);results.append(result)
    write(ROOT/'state/P_only_profile_complete.json',dict(time=time.time(),complete=True,
        observer_passed=[r['observer_gate_passed'] for r in results]))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/P_only_profile_hold.json',dict(time=time.time(),error=repr(error)));raise
