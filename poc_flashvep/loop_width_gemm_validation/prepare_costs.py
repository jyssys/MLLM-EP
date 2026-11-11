"""Prepare serial path costs and a larger Token execution group."""
import copy
from .common import *


def main():
    initialize()
    extra=[]
    for count in (8,32):
        cfg=json.loads((ROOT/'configs'/f'kernel_huginn_token_state{count}.json').read_text())
        cfg.update(name=f'kernel_huginn_token_group32_state{count}',D_prefill_loop_cap=32,
            phase='LARGE_TOKEN_EXECUTION_GROUP_EXACT_STATE_QUALIFICATION',
            required_qualification='kernel_huginn_token_state32' if count==8 else 'kernel_huginn_token_group32_state8')
        save(cfg);extra.append(cfg['name'])
    serial=[]
    for model in ('huginn','ouro'):
        arms=('zero','loop','token','token_group32') if model=='huginn' else ('zero','loop','token')
        for arm in arms:
            base=json.loads((ROOT/'configs'/f'kernel_{model}_{arm}_state8.json').read_text())
            for policy in ('original','tuned'):
                cfg=copy.deepcopy(base)
                cfg.update(name=f'cost_{model}_{arm}_{policy}_serial8_s1',state_probe=False,
                    verify_transfer=False,async_prompt_state_probe=False,async_verify_handoff=False,
                    calibration_serial=True,observe_device_execution=True,observe_service_timing=True,
                    phase='CUDA_EVENT_SERIAL_PATH_CALIBRATION_NOT_ONLINE_SLACK',
                    required_qualification=f'kernel_{model}_{arm}_state32',load='serial',
                    projection_kernel_policy='tuned_s1' if policy=='tuned' else 'off',
                    clean_confirmation=False,arrival_pattern='SERIAL_CALIBRATION',
                    observe_P_submission_readiness=False,host_cost_trace=False)
                save(cfg);serial.append(cfg['name'])
    write(ROOT/'state/serial_cost_plan.json',dict(qualification=extra,jobs=serial,
        scope='Actual Native-path serial requests. CUDA events enclose stream waits and launch gaps. Not clean serving or exclusive kernels.'))
    print(' '.join(extra+serial))


if __name__=='__main__':main()
