"""Calibrate realized paced and grouped Token execution after v2 qualification."""
from .common import *


def main():
    initialize();jobs=[]
    for arm in ('zero','loop','token','token_group32','token1792'):
        cfg=json.loads((ROOT/'configs'/f'kernel_v2_huginn_{arm}_state8.json').read_text())
        cfg.update(name=f'cost_v2_huginn_{arm}_tuned_serial8_s1',state_probe=False,verify_transfer=False,
            async_prompt_state_probe=False,async_verify_handoff=False,calibration_serial=True,
            observe_device_execution=True,observe_service_timing=True,observe_core_graph_instance=True,
            phase='REALIZED_V2_QUANTUM_SERIAL_CALIBRATION_NOT_SERVING',
            required_qualification=f'kernel_v2_huginn_{arm}_state32',clean_confirmation=False,
            arrival_pattern='SERIAL_CALIBRATION',load='serial',observe_P_submission_readiness=False,
            serial_output_reference='cost_huginn_zero_original_serial8_s1',
            serial_baseline='cost_v2_huginn_zero_tuned_serial8_s1')
        save(cfg);jobs.append(cfg['name'])
    write(ROOT/'state/v2_serial_quantum_plan.json',dict(jobs=jobs,
        scope='Fresh state-qualified v2 policy. Token cap includes Prelude. Actual quantum audit is mandatory.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
