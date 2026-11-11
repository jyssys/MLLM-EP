"""User-priority matched-row comparisons. No online Serving conclusion."""
import copy,json
from .common import *


def main():
    initialize();jobs=[];matrix=[]
    groups=[('ouro',1024,3),('huginn',128,30),('huginn',256,28),('huginn',512,24)]
    baselines={}
    for model,width,cut in groups:
        length=4096 if model=='ouro' else 2048;depth=4 if model=='ouro' else 32
        assert width*depth==length*(depth-cut)
        state_names={}
        for arm in ('loop','token'):
            source=WIDTH/'configs'/f'kernel_v2_{model}_{"token1792" if arm=="token" and model=="huginn" else arm}_state8.json'
            cfg=json.loads(source.read_text());name=f'priority_width_{model}_S{width}_{arm}_grouped_state8'
            cfg.update(name=name,native_partial_cut=cut if arm=='loop' else None,
                native_frontier=[0,length-width] if arm=='token' else None,cut=cut if arm=='loop' else depth,
                method=arm,axis='loop' if arm=='loop' else 'token',D_prefill_loop_cap=depth,
                D_prefill_cap_includes_prelude=True,D_prefill_token_cap=width if arm=='token' else None,
                phase='USER_PRIORITY_GROUPED_MATCHED_ROW_EXACT_STATE_CHECK',
                discovery_activity=False,discovery_event_observer=False,priority_serial_comparison=False,
                required_qualification=f'native_{model}_{"early_profile" if model=="ouro" else "clean_recovery"}_state32',
                timeout=900,startup_timeout=300,require_realized_quantum_contract=True)
            save(cfg);jobs.append(name);state_names[arm]=name
        if model not in baselines:
            source=WIDTH/'configs'/f'kernel_v2_{model}_zero_state8.json'
            cfg=json.loads(source.read_text());name=f'priority_width_{model}_zero_tuned_serial8_s1'
            cfg.update(name=name,state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
                async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
                observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
                phase='USER_PRIORITY_SERIAL_CALIBRATION_NOT_SERVING',load='serial',arrival_pattern='SERIAL_CALIBRATION',
                required_qualification=f'native_{model}_{"early_profile" if model=="ouro" else "clean_recovery"}_state32',
                clean_confirmation=False,observe_P_submission_readiness=False,host_cost_trace=False,
                discovery_activity=False,discovery_event_observer=False,serial_output_reference=name,serial_baseline=name)
            save(cfg);jobs.append(name);baselines[model]=name
        for arm in ('loop','token'):
            cfg=json.loads((ROOT/'configs'/(state_names[arm]+'.json')).read_text());name=f'priority_width_{model}_S{width}_{arm}_grouped_tuned_serial8_s1'
            cfg.update(name=name,state_probe=False,verify_transfer=False,async_prompt_state_probe=False,
                async_verify_handoff=False,calibration_serial=True,observe_device_execution=True,
                observe_service_timing=True,observe_core_graph_instance=True,priority_serial_comparison=True,
                phase='USER_PRIORITY_SERIAL_CALIBRATION_NOT_SERVING',load='serial',arrival_pattern='SERIAL_CALIBRATION',
                required_qualification=state_names[arm],clean_confirmation=False,
                observe_P_submission_readiness=False,host_cost_trace=False,
                serial_output_reference=baselines[model],serial_baseline=baselines[model])
            save(cfg);jobs.append(name)
        matrix.append(dict(model=model,length=length,depth=depth,token_suffix=width,loop_cut=cut,
            D_loops=depth-cut,recurrent_rows_per_request=width*depth,
            token_group_limit=depth,loop_group_limit=depth,
            scope='Both arms can group the full remaining work. Token is not artificially restricted to one-loop host submissions. Equal recurrent rows do not imply equal Attention or boundary work.'))
    write(ROOT/'state/priority_width_comparison_plan.json',dict(jobs=jobs,matrix=matrix,
        requested_before='Resume the marker-free workload discovery sequence',
        scope='N8 full-state qualification precedes each fresh serial calibration. No new adaptive scheduler or online Decode-safe claim.'))
    print(' '.join(jobs))


if __name__=='__main__':main()
