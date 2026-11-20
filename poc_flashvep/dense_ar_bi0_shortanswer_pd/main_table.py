"""Keep performance values and per-job provenance in the same table."""
import csv
import json
from .common import ROOT,MODEL_REVISION,digest,write


def main():
    patterns=('Main_*_ShortAnswer200_*_BI0_v3_metrics.json',
        'ControlledDiscovery_*_BI0_v3_metrics.json','Confirmation_*_BI0_v2_metrics.json',
        'FixedQPS_*_BI0_v1_metrics.json','EqualOutputControl_*_BI0_v1_metrics.json','RootCut12_*_BI0_v1_metrics.json','LowQPSControl_*_BI0_v1_metrics.json')
    files=sorted({p for pattern in patterns for p in (ROOT/'analysis').glob(pattern)})
    rows=[]
    for file in files:
        metrics=json.loads(file.read_text());name=metrics['name'];folder=ROOT/'raw'/name
        cfg=json.loads((folder/'configuration.json').read_text())
        receipt=json.loads((folder/'dispatcher_result.json').read_text())
        quality=json.loads((ROOT/'analysis'/f'{name}_quality_gate.json').read_text())
        runtime={p.stem.removeprefix('actual_runtime_'):json.loads(p.read_text()) for p in folder.glob('actual_runtime_*.json')}
        trace=ROOT/'analysis'/f'{name}_timeline_analysis.json'
        geometry=json.loads(trace.read_text()).get('geometry',{}) if trace.exists() else {}
        axis=ROOT/'analysis'/f'{name}_axis_timeline.json'
        accounting=json.loads(axis.read_text()) if axis.exists() else None
        migrated=[r for r in accounting['rows'] if r['action']['cut']>0] if accounting else None
        offloaded_rows=sum(r['D_token_layer_rows'] for r in migrated) if migrated is not None else None
        row=dict(name=name,method=cfg['method'],phase=cfg.get('phase'),N=metrics['n'],
            Output_contract='DIAGNOSTIC_FIXED_LENGTH_REPLAY' if cfg.get('diagnostic_controlled_output_replay') else 'NATURAL_EOS',
            B=cfg['token_budget'],S=cfg['max_num_seqs'],KV_GiB=42,KV_block_tokens=cfg.get('KV_block_tokens',16),
            Model_revision=MODEL_REVISION,Dataset_SHA256=cfg['dataset_SHA256'],
            Source_archive_SHA256=receipt['source_SHA256'],Config_SHA256=digest(folder/'configuration.json'),
            Arrival_seed=cfg.get('arrival_seed'),Controlled_pipeline=cfg.get('controlled_pipeline',False),
            Actual_batch_histograms={role:{key:value[key] for key in ('actual_request_batch_histogram',
                'actual_prefill_request_batch_histogram','actual_decode_request_batch_histogram','mixed_prefill_decode_iterations')} for role,value in geometry.items()},
            Actual_query_geometry={role:{key:value[key] for key in ('valid_query_rows','padded_query_rows','actual_query_lengths','measured_batches')} for role,value in geometry.items()},
            Actual_worker_configuration=runtime,Declared_GPUs=cfg['gpus'],
            Axis_cut=cfg.get('axis_cut',0),Source_credit_limit=cfg.get('source_credit_limit'),
            Migration_fraction=len(migrated)/len(accounting['rows']) if migrated is not None else 0. if not cfg.get('axis_cut') else None,
            Actual_migrated_requests=len(migrated) if migrated is not None else None,
            Actual_fallback_requests=len(accounting['rows'])-len(migrated) if migrated is not None and cfg.get('axis_cut') else 0,
            Actual_offloaded_prompt_GEMM_rows=offloaded_rows,
            Offloaded_prompt_GEMM_rows_per_second=offloaded_rows/metrics['measurement_seconds'] if offloaded_rows is not None else None,
            P_token_layer_rows=sum(r['P_token_layer_rows'] for r in accounting['rows']) if accounting else None,
            D_token_layer_rows=sum(r['D_token_layer_rows'] for r in accounting['rows']) if accounting else None,
            Prompt_work_coverage=accounting['all_prompt_work_accounted'] if accounting else 'NOT_REANALYZED',
            Task_quality_point=quality['point_passed'],Task_quality_CI=quality['CI_passed'],
            Strict_Greedy_matches_to_C1_reference=quality['strict_tokens_matched'],
            Strict_reference_requests=quality['n'],
            Accuracy_contract='BI0 cross-geometry task-quality CI. Not strict whole200 greedy equality.',
            Full_state_capture_in_this_clean_run=False,
            Qualified_Natural_measurement=bool(receipt['complete'] and quality['CI_passed'] and metrics['n']==200
                and not cfg.get('diagnostic_controlled_output_replay')),
            Kernel_measurement_scope='NOT_MEASURED_PRIMARY_IN_THIS_CLEAN_RUN',
            Host_measurement_scope=metrics['timing_scope'],
            Sustainable_capacity=False,Independent_starts=1,
            evidence='DIAGNOSTIC' if cfg.get('diagnostic_controlled_output_replay') else 'MEASURED_ONLINE')
        for key in ('offered_QPS','actual_arrival_QPS','actual_dispatch_QPS','completed_requests_per_second',
            'TTFT_mean_ms','TTFT_p50_ms','TTFT_p95_ms','TTFT_p99_ms','mean_TBT_ms','ITL_p99_ms',
            'max_ITL_ms','TPOT_p95_ms','E2E_p95_ms','output_tokens','generated_output_tokens_per_second',
            'SLO_qualified_output_tokens_per_second','Joint_SLO_attainment','measurement_seconds'):
            row[key]=metrics[key]
        rows.append(row)
    value=dict(rows=rows,evidence='Each row declares its scope.',
        scope='One row per independent server start. Failed-quality cells remain visible. '
        'Natural and fixed-output replay results are separate. Actual worker settings and source ZIP hashes are retained. '
        'The table does not by itself qualify native-relative decode noninferiority or sustainable capacity.')
    write(ROOT/'analysis/BI0_main_table_with_provenance.json',value)
    if rows:
        with (ROOT/'analysis/BI0_main_table_with_provenance.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:
                writer.writerow({key:json.dumps(item,separators=(',',':')) if isinstance(item,(list,dict)) else item for key,item in row.items()})
    return value


if __name__=='__main__':main()
