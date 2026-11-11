"""Add matched-row larger cuts before serial measurement starts."""
import copy
from .common import *


def main():
    initialize();plan_path=ROOT/'state/serial_cost_plan.json'
    plan=json.loads(plan_path.read_text())
    assert plan.get('revision',1)==1,'The larger-cut plan is already immutable'
    assert not any((ROOT/'raw'/name).exists() for name in plan['qualification']+plan['jobs'])
    write(ROOT/'state/serial_cost_plan_v1.json',plan)
    for arm,options in (
        ('loop28',dict(native_partial_cut=28,cut=28,D_prefill_loop_cap=1)),
        ('token1792',dict(native_frontier=[0,1792],D_prefill_loop_cap=1))):
        source='loop' if arm=='loop28' else 'token'
        for count in (8,32):
            cfg=json.loads((ROOT/'configs'/f'kernel_huginn_{source}_state{count}.json').read_text())
            cfg.update(name=f'kernel_huginn_{arm}_state{count}',**options,
                phase='LARGER_EQUAL_RECURRENT_ROW_CUT_QUALIFICATION',
                required_qualification=f'kernel_huginn_{source}_state32' if count==8 else f'kernel_huginn_{arm}_state8')
            save(cfg);plan['qualification'].append(cfg['name'])
        base=json.loads((ROOT/'configs'/f'kernel_huginn_{arm}_state8.json').read_text())
        for policy in ('original','tuned'):
            cfg=copy.deepcopy(base)
            cfg.update(name=f'cost_huginn_{arm}_{policy}_serial8_s1',state_probe=False,
                verify_transfer=False,async_prompt_state_probe=False,async_verify_handoff=False,
                calibration_serial=True,observe_device_execution=True,observe_service_timing=True,
                phase='CUDA_EVENT_SERIAL_PATH_CALIBRATION_NOT_ONLINE_SLACK',
                required_qualification=f'kernel_huginn_{arm}_state32',load='serial',
                projection_kernel_policy='tuned_s1' if policy=='tuned' else 'off',
                clean_confirmation=False,arrival_pattern='SERIAL_CALIBRATION',
                observe_P_submission_readiness=False,host_cost_trace=False)
            save(cfg);plan['jobs'].append(cfg['name'])
    plan['revision']=2;write(plan_path,plan)
    print('QUALIFICATION',len(plan['qualification']),'SERIAL',len(plan['jobs']))


if __name__=='__main__':main()
