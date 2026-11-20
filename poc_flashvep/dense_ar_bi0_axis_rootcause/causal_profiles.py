"""Profile source kernels only after the external-ID witness passes."""
import json
import time
from .common import ROOT,write
from .campaign import cell


def main():
    design=ROOT/'configs/frozen_causal_P_profile_v1.json'
    if not design.exists():
        write(design,dict(time=time.time(),roles=['prefill'],cut=4,N=200,seed=2026112082,QPS=2.35,
            modes=['SERIAL_ADMISSION','ONLINE'],axes=['Layer','Token'],output='Fixed C1 request lengths, total4723tokens',
            observer_fields=['TTFT_mean_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','P_prepare_to_ready_mean_ms'],
            observer_limit=.05,controls=2,profiled_starts=1,
            scope='P-only kernel correlation. D suffix and decode remain separate host enclosures. No failed observer is promoted to primary kernel evidence.'))
    while not (ROOT/'state/serial_online_causal_control_complete.json').exists() or not (ROOT/'state/correlated_collector_witness_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    witness=json.loads((ROOT/'analysis/correlated_collector_witness.json').read_text())
    if not witness['eager_correlation_passed']:
        write(ROOT/'state/causal_P_profiles_not_run.json',dict(time=time.time(),status='NOT_RUN',reason='External kernel-correlation witness failed.'))
        return
    base=json.loads((ROOT/'analysis/serial_online_causal_control_v1.json').read_text())['results']
    results=[]
    for axis in ('Layer','Token'):
        for mode in ('SERIAL_ADMISSION','ONLINE'):
            original=next(r for r in base if r['axis']==axis and r['causal_mode']==mode and r['credit']==4)
            cfg=json.loads((ROOT/'raw'/original['name']/'configuration.json').read_text());names=dict(clean_before=original['name'])
            for label in ('profile','clean_after'):
                name=f'CausalPProfile_{axis}_{mode}_{label}_s2026112082_N200_v1';names[label]=name
                row=cell(name,axis,4,seed=2026112082,
                    entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',strict_serial_pipeline=mode=='SERIAL_ADMISSION',
                    diagnostic_controlled_output_replay=True,replay_output_lengths=cfg['replay_output_lengths'],
                    worker_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_worker',
                    kernel_activity=label=='profile',correlated_profile=label=='profile',
                    kernel_activity_roles=['prefill'],kernel_activity_seconds=7200,activity_copies=True,
                    phase='CAUSAL_P_KERNEL_OBSERVER_CONTROL')
                assert row['status']!='FAILED',name
            results.append(dict(axis=axis,mode=mode,names=names))
            write(ROOT/'analysis/causal_P_profile_triplets.json',dict(results=results,
                scope='Same causal contract. Profile intervention includes the correlation collector and its thin host wrappers. Original clean-before and the new clean-after retain the unmodified model path.'))
    write(ROOT/'state/causal_P_profiles_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/causal_P_profiles_hold.json',dict(time=time.time(),error=repr(error)));raise
