"""Recalculate policy discoveries without treating requests as server starts."""
import json
import time
from .common import ROOT,write
from .metrics import summarize


PREFIXES=('GuardAB_Discovery200_','GuardTune_Discovery200_',
    'ThrottleAB_Discovery200_','ComputeTune_Discovery200_',
    'RemoteSetupAB_Discovery200_','RemoteTune_Discovery200_',
    'DcapTune_Discovery200_','FractionalTune_Discovery200_',
    'WholeDeflect_Discovery200_','DeflectionAB_Discovery200_','NativeCompute_Discovery200_',
    'CreditTune_Discovery200_','CompletionAB_Discovery200_','CompoundTune_Discovery200_')


def main():
    rows=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.name.startswith(PREFIXES):
            continue
        result=folder/'dispatcher_result.json'
        if not result.exists():
            continue
        status=json.loads(result.read_text())
        if not status['complete']:
            rows.append(dict(name=folder.name,status='FAILED',result=status))
            continue
        cfg=json.loads((folder/'configuration.json').read_text())
        metric=summarize(folder.name)
        fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms',
            'SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second',
            'Joint_SLO_attainment','completed_requests_per_second')
        quality_path=ROOT/'analysis'/(folder.name+'_quality_gate.json')
        score=json.loads(quality_path.read_text()) if quality_path.exists() else None
        rows.append(dict(name=folder.name,status='MEASURED_ONLINE',independent_starts=1,
            QPS=cfg['qps'],axis=cfg.get('axis_kind','Native'),cut=cfg.get('axis_cut',0),
            suffix_fraction=cfg.get('hybrid_suffix_fraction'),
            fractional_token_work=cfg.get('fractional_token_work'),
            D_cap=cfg.get('D_running_admission_cap'),
            D_quiescent=cfg.get('d_quiescent_admission',False),
            D_compute_deferral=cfg.get('d_compute_throttle',False),
            D_READ_setup_deferral=cfg.get('defer_D_remote_setup',False),
            compiled=cfg['execution_family'],Source_credit=cfg.get('continuous_source_credit_limit',0),
            GPU_pair=cfg['gpus'],dataset_SHA256=cfg['dataset_SHA256'],seed=cfg['seed'],
            source_archive_SHA256=status['source_SHA256'],
            actual_geometry=metric.get('actual_geometry'),
            task_quality_passed=score.get('passed') if score else None,
            metrics={k:metric.get(k) for k in fields}))
    comparisons=[]
    by_name={r['name']:r for r in rows if r['status']=='MEASURED_ONLINE'}
    for qps in (1.8,2.8):
        for axis in ('Native','Layer','Token','Hybrid'):
            left=f'RemoteSetupAB_Discovery200_q{qps}_{axis}'+('Setup_v1')
            right=f'RemoteSetupAB_Discovery200_q{qps}_{axis}'+('Strong_v1' if axis=='Native' else 'Throttle_v1')
            if left not in by_name or right not in by_name:
                continue
            a,b=by_name[left],by_name[right]
            assert all(a[k]==b[k] for k in ('GPU_pair','dataset_SHA256','seed','source_archive_SHA256'))
            comparisons.append(dict(READ_setup_arm=left,control=right,
                ratio={k:a['metrics'][k]/b['metrics'][k] for k in
                    ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second')},
                independent_paired_starts=1,
                scope='Within-axis same-source/pair/trace policy intervention. '
                    'The Native comparison also enables the compute rule,whose last-query exemption remains unchanged. '
                    'No exclusive host/GPU time attribution and no confidence-interval safety claim.'))
    value=dict(time=time.time(),rows=rows,READ_setup_comparisons=comparisons,evidence='MEASURED_ONLINE',
        scope='All completed discovery outcomes remain visible. Metrics use unchanged client timestamps and SLO. '
            'Each row is one independent server start. Do not infer a final non-inferiority CI. '
            'Different families may use different compile/source admission contracts. '
            'READ deferral and running caps are explicit common-policy candidates,not unmodified default vLLM. '
            'Finite-cohort throughput is not sustainable capacity.')
    write(ROOT/'analysis/decode_tail_recovery_discovery_all_v1.json',value)
    print('Completed recovery discovery rows',len(rows),'paired READ interventions',len(comparisons),flush=True)


if __name__=='__main__':
    main()
