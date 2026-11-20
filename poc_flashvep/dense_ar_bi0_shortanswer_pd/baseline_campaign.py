"""Measure frozen BI0 common baselines and a separate transport intervention."""
import json
import time
from .common import ROOT, write
from .campaign import serving
from .serving_analysis import analyze
from .online_profile import summarize_triplet


def main():
    assert (ROOT/'state/NIXL_thread_screen_complete.json').exists()
    contract=ROOT/'configs/frozen_BI0_baseline_design_v1.json'
    if not contract.exists():
        write(contract,dict(time=time.time(),dataset='longbench_shortanswer200_v2.json',count=200,
            QPS=[.4,1.2,2.35],arrival_seeds=[2026112051,2026112052,2026112053],
            common_methods=['Native','Agg'],common_NIXL_threads=4,
            confirmation_QPS=2.35,transport_intervention=dict(num_threads=1,QPS=2.35,starts=3),
            profile=dict(QPS=2.35,seconds=90,order=['clean_before','profile','clean_after']),
            selection='N32 screen motivated a one-thread diagnostic. Main common baseline remains the official four-thread default.',
            gates='Report full200 task quality before qualification. Do not infer safe windows from kernel inactivity.',
            scope='Finite cohorts. Discovery seed51. Confirmation seeds52/53. Natural output work is reported for every arm.'))
    design=json.loads(contract.read_text());results=[]
    for rate in design['QPS']:
        seeds=design['arrival_seeds'] if rate==design['confirmation_QPS'] else design['arrival_seeds'][:1]
        for seed in seeds:
            for method in design['common_methods']:
                name=f'Main_{method}_ShortAnswer200_q{rate:g}_s{seed}_BI0_v1'
                result=serving(name,method,design['dataset'],200,rate,seed,phase='COMMON_BI0_BASELINE')
                analyze(name);results.append(result)
                write(ROOT/'analysis/BI0_baseline_results_v1.json',dict(results=results))
    write(ROOT/'state/main_common_sweep_complete.json',dict(time=time.time(),complete=True))
    for seed in design['arrival_seeds']:
        name=f'Diagnostic_Native_T1_ShortAnswer200_q2.35_s{seed}_BI0_v1'
        result=serving(name,'Native',design['dataset'],200,2.35,seed,
            nixl_num_threads=1,phase='NIXL_ONE_THREAD_INTERVENTION')
        analyze(name);results.append(result)
        write(ROOT/'analysis/BI0_baseline_results_v1.json',dict(results=results))
    names={}
    for label in design['profile']['order']:
        name=f'Profile_Native_ShortAnswer200_q2.35_{label}_BI0_v1';names[label]=name
        serving(name,'Native',design['dataset'],200,2.35,design['arrival_seeds'][0],
            phase='KERNEL_ACTIVITY_OBSERVER_CONTROL',kernel_activity=label=='profile',
            kernel_activity_seconds=design['profile']['seconds'],activity_copies=True,quiescent_witness=True)
        analyze(name)
    result=summarize_triplet(2.35,names)
    write(ROOT/'state/BI0_baseline_campaign_complete.json',dict(time=time.time(),complete=True,
        observer_passed=result['observer_gate_passed']))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/BI0_baseline_campaign_hold.json',dict(time=time.time(),error=repr(error)))
        raise
