"""Repair an identified attribution defect. Keep the failed profiles visible."""
import json
import time
from .common import ROOT,write,save
from .run import execute
from .campaign import cell


def main():
    declaration=ROOT/'configs/frozen_runtime_api_profile_v1.json'
    if not declaration.exists():
        write(declaration,dict(time=time.time(),reason='The DRIVER-only collector mapped58.22% of source kernel time. CUDA Runtime-launched attention and norm kernels had no external API mapping.',
            change='Enable the existing RUNTIME plus DRIVER collector mode. No model or scheduler change.',
            observer_fields=['TTFT_mean_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms','P_prepare_to_ready_mean_ms'],
            observer_limit=.05,attribution_minimum=.99,cut=4,N=200,
            sources='Fixed-output diagnostic seed2082 and original Natural EOS seed2061, same200 inputs.',
            rule='Do not select runs by timing. A failed repaired profile remains diagnostic. Old raw records stay unchanged.',
            reference='https://docs.nvidia.com/cupti/main/main.html#correlation-id-mechanism'))
    cfg=dict(name='RuntimeAPICorrelation_Witness_v1',method='Native',
        entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.correlation_witness',
        gpus=[1],batch_invariant=False,correctness_only=True,wall_timeout=180,runtime_api_correlation=True)
    assert execute(save(cfg))
    write(ROOT/'state/runtime_api_correlation_witness_complete.json',dict(time=time.time(),passed=True))
    while not (ROOT/'state/natural_Last4_reproduction_complete.json').exists() or not (ROOT/'state/causal_P_profiles_complete.json').exists():
        assert not (ROOT/'state/STOP_REQUESTED.json').exists();time.sleep(5)
    declarations=[]
    for axis in ('Layer','Token'):
        for mode in ('ONLINE','SERIAL_ADMISSION','NATURAL_REPRODUCTION'):
            natural=mode=='NATURAL_REPRODUCTION'
            source_path=ROOT/'analysis'/(f'natural_Last4_{axis}_profile_triplet.json' if natural else 'causal_P_profile_triplets.json')
            raw=json.loads(source_path.read_text())
            prior=raw if natural else next(r for r in raw['results'] if r['axis']==axis and r['mode']==mode)
            before=prior['names']['clean_after'];configuration=json.loads((ROOT/'raw'/before/'configuration.json').read_text())
            seed=2026112061 if natural else 2026112082
            names=dict(clean_before=before)
            for label in ('profile','clean_after'):
                name=f'RuntimeAPIProfile_{axis}_{mode}_{label}_s{seed}_N200_v1';names[label]=name
                extra={} if natural else dict(diagnostic_controlled_output_replay=True,replay_output_lengths=configuration['replay_output_lengths'])
                row=cell(name,axis,4,seed=seed,
                    entry_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_serving',
                    strict_serial_pipeline=mode=='SERIAL_ADMISSION',worker_module='poc_flashvep.dense_ar_bi0_axis_rootcause.causal_worker',
                    kernel_activity=label=='profile',correlated_profile=label=='profile',runtime_api_correlation=True,
                    kernel_activity_roles=['prefill'],kernel_activity_seconds=7200,activity_copies=True,
                    phase='RUNTIME_API_ATTRIBUTION_REPAIR',**extra)
                assert row['status']!='FAILED',name
            declarations.append(dict(axis=axis,mode=mode,names=names))
            write(ROOT/'analysis/runtime_api_profile_triplets.json',dict(results=declarations))
    write(ROOT/'state/runtime_api_profiles_complete.json',dict(time=time.time(),complete=True))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'state/runtime_api_profiles_hold.json',dict(time=time.time(),error=repr(error)));raise
