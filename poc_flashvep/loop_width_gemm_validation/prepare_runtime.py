import copy,shutil
from .common import *

FROZEN=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_intervention_dynamic_loop'


def main():
    initialize();policy=ROOT/'configs/operator_kernel_policy_s1.json';assert policy.exists()
    shutil.copy2(OLD_ROOT/'configs/frozen_SLO.json',ROOT/'configs/frozen_SLO.json')
    for model,length in (('huginn',2048),('ouro',4096)):
        origin=OLD_ROOT/'data'/f'heldout_{model}_L{length}.json'
        if not origin.exists():origin=FROZEN/'data'/f'heldout_{model}_L{length}.json'
        shutil.copy2(origin,ROOT/'data'/origin.name)
    jobs=[]
    for count in (8,32):
        for model in ('huginn','ouro'):
            for arm in ('zero','loop','token'):
                if model=='huginn':
                    parent={'zero':'lifetime3_deferred2_zero','loop':'priority4_normal_loop31','token':'quantum6_token1984_1'}[arm]+f'_state{count}'
                    path=OLD_ROOT/'configs'/(parent+'.json')
                else:
                    path=FROZEN/'configs/n32_ouro_native_full_state32.json'
                cfg=json.loads(path.read_text());name=f'kernel_{model}_{arm}_state{count}'
                cfg.update(name=name,count=count,request_caps=cfg['request_caps'][:count],
                    prompt_ids=cfg['prompt_ids'][:count],arrival_offsets=[0.]*count,
                    phase='TUNED_KERNEL_EXACT_STATE_QUALIFICATION_NOT_TIMING',required_qualification=None,
                    timeout=900,startup_timeout=300,projection_kernel_policy='tuned_s1',
                    projection_kernel_policy_file=str(policy),projection_kernel_policy_SHA256=digest(policy),
                    clean_confirmation=False,width_parent_configuration=dict(path=str(path),sha256=digest(path)),
                    observe_service_timing=False,sparse_device_trace=False,host_cost_trace=False)
                if count==32:cfg['required_qualification']=f'kernel_{model}_{arm}_state8'
                elif arm!='zero':cfg['required_qualification']=f'kernel_{model}_zero_state8'
                if model=='ouro':
                    cfg.update(atomic_prompt=arm=='zero',native_path=True,feature_native_wire_v5=True,
                        native_partial_cut=3 if arm=='loop' else None,native_frontier=[0,3072] if arm=='token' else None,
                        cut=3 if arm=='loop' else 4,method='native' if arm=='zero' else arm,axis='token' if arm=='token' else 'loop',
                        intervention_method='native',dlocal_campaign=False,D_local_probe_output_pacing=False,
                        state_probe=True,verify_transfer=True,deep_graph_policy='primary_reserved',deep_graph_budget=64,
                        D_prefill_loop_cap=1 if arm!='zero' else None,
                        D_prefill_token_cap=1024 if arm=='token' else None,
                        explicit_graph_teardown=True,child_exit_integrity_required=True)
                save(cfg);jobs.append(name)
    write(ROOT/'state/runtime_qualification_plan.json',dict(jobs=jobs,
        scope='Same frozen input/seed/state references. N8 precedes N32. No serving success follows from operator timing.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
