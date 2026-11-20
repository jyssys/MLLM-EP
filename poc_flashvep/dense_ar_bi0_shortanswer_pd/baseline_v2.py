"""Restore official UCX initialization before final baseline qualification."""
import json
import time
from .common import ROOT,write,save
from .qualification import configuration
from .run import execute
from .campaign import serving
from .serving_analysis import analyze
from .online_profile import summarize_triplet


def preflight():
    name='qual_Native_OfficialInit_ShortAnswer_N8_BI0_v2'
    cfg=configuration(name,'Native','qualification8.json',restore_official_UCX_init=True)
    assert execute(save(cfg)),name
    folder=ROOT/'raw'/name
    teacher=json.loads((ROOT/'raw/qual_Standard_ShortAnswer_N8_BI0_state_v1/prompt_state_single.json').read_text())['requests']
    p=json.loads((folder/'prompt_state_prefill.json').read_text())['requests']
    d=json.loads((folder/'prompt_state_decode.json').read_text())['requests']
    rows=[dict(request_id=rid,KV_exact=teacher[rid]['KV']==p[rid]['KV']==d[rid]['KV'],
        P_hidden_exact=teacher[rid]['final_hidden_SHA256']==p[rid]['final_hidden_SHA256']) for rid in teacher if rid.startswith('q')]
    passed=len(rows)==8 and all(r['KV_exact'] and r['P_hidden_exact'] for r in rows)
    for role in ('prefill','decode'):
        actual=json.loads((folder/f'actual_runtime_{role}.json').read_text())
        assert actual['UCX_RCACHE_MAX_UNRELEASED']=='1024'
    write(ROOT/'analysis/BI0_official_UCX_init_N8_gate_v2.json',dict(passed=passed,rows=rows,evidence='CORRECTNESS_ONLY'))
    assert passed


def main(tag='v2'):
    declaration=ROOT/'configs'/f'frozen_BI0_baseline_design_{tag}.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),dataset='longbench_shortanswer200_v2.json',N=200,QPS=[.4,1.2,2.35],
            seeds=[2026112051,2026112052,2026112053],primary_UCX_RCACHE_MAX_UNRELEASED='1024',
            B=8192,S=1024,default_KV_block_tokens=16,NIXL_threads=4,
            reason='Observer imported SDK before the official default setup. Preserve v1 as its exact historical contract. '
            'Correct initialization is required irrespective of the A/B performance effect.',
            scopes=['Same-trace N32 initialization A/B','Full200 common primary','Separate one-thread diagnostic','Observer-controlled kernel activity'],
            quality='Full200 point-quality failure stops dispatch. CI qualification is reported separately.'))
    preflight()
    for label,restore in (('ObserverImport',False),('OfficialInit',True)):
        name=f'InitAB_{label}_Native_N32_q2.35_BI0_v2'
        serving(name,'Native','qualification32.json',32,2.35,2026112033,
            restore_official_UCX_init=restore,phase='UCX_INITIALIZATION_DIAGNOSTIC')
        analyze(name)
    results=[]
    for rate in (.4,1.2,2.35):
        seeds=(2026112051,2026112052,2026112053) if rate==2.35 else (2026112051,)
        for seed in seeds:
            for method in ('Native','Agg'):
                name=f'Main_{method}_ShortAnswer200_q{rate:g}_s{seed}_BI0_{tag}'
                result=serving(name,method,'longbench_shortanswer200_v2.json',200,rate,seed,
                    restore_official_UCX_init=True,phase='COMMON_BI0_OFFICIAL_INIT_BASELINE')
                analyze(name);results.append(result)
                write(ROOT/'analysis'/f'BI0_baseline_results_{tag}.json',dict(results=results))
                assert result['quality_gate']['point_passed'],name
    write(ROOT/'state'/f'main_common_sweep_{tag}_complete.json',dict(time=time.time(),complete=True))
    for seed in (2026112051,2026112052,2026112053):
        name=f'Diagnostic_Native_T1_ShortAnswer200_q2.35_s{seed}_BI0_{tag}'
        result=serving(name,'Native','longbench_shortanswer200_v2.json',200,2.35,seed,
            restore_official_UCX_init=True,nixl_num_threads=1,phase='OFFICIAL_INIT_NIXL_ONE_THREAD_INTERVENTION')
        analyze(name);results.append(result)
        write(ROOT/'analysis'/f'BI0_baseline_results_{tag}.json',dict(results=results))
    names={}
    for label in ('clean_before','profile','clean_after'):
        name=f'Profile_Native_ShortAnswer200_q2.35_{label}_BI0_{tag}';names[label]=name
        serving(name,'Native','longbench_shortanswer200_v2.json',200,2.35,2026112051,
            restore_official_UCX_init=True,phase='OFFICIAL_INIT_KERNEL_ACTIVITY',kernel_activity=label=='profile',
            kernel_activity_seconds=90,activity_copies=True,quiescent_witness=True)
        analyze(name)
    profile=summarize_triplet(2.35,names)
    write(ROOT/'analysis'/f'BI0_official_init_online_observer_{tag}.json',profile)
    write(ROOT/'state'/f'BI0_baseline_{tag}_complete.json',dict(time=time.time(),complete=True,
        observer_passed=profile['observer_gate_passed']))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/baseline_v2_hold.json',dict(time=time.time(),error=repr(error)));raise
