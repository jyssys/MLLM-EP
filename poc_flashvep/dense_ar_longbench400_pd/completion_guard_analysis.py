"""Test common admission effects without calling host waits GPU time."""
import json
import time
from .common import ROOT,write
from .metrics import summarize
from .quality_gate import compare as quality
from .chunk_residency_audit import analyze as chunks
from .service_accounting import compare as boundaries


def main():
    migrated=ROOT/'state/completion_guard_resource_migration_v2.json'
    mapping=json.loads(migrated.read_text())['old_to_new'] if migrated.exists() else {}
    cells=[]
    for qps in (1.8,2.8):
        for variant in ('Native','Layer','StockToken','Hybrid'):
            names=[f'CompletionAB_Discovery200_q{qps}_{variant}_Guard{i}_v1' for i in (0,1)]
            names=[mapping.get(n,n) for n in names]
            row=dict(QPS=qps,variant=variant,guard0=names[0],guard1=names[1],status='NOT_RUN')
            receipt=[ROOT/'raw'/name/'dispatcher_result.json' for name in names]
            if not all(p.exists() for p in receipt):
                cells.append(row)
                continue
            if not all(json.loads(p.read_text())['complete'] for p in receipt):
                row['status']='FAILED'
                cells.append(row)
                continue
            config=[json.loads((ROOT/'raw'/n/'configuration.json').read_text()) for n in names]
            for field in ('dataset_SHA256','arrival_offsets','gpus','seed','axis_cut',
                    'continuous_source_credit_limit','token_budget','max_num_seqs','source_capsule_SHA256'):
                assert config[0].get(field)==config[1].get(field),(names,field)
            assert not config[0]['D_prefill_completion_guard'] and config[1]['D_prefill_completion_guard']
            scores=[quality(n) for n in names]
            metrics=[summarize(n) for n in names]
            chunk=[chunks(n) for n in names]
            fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms',
                'SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second')
            row.update(status='MEASURED_ONLINE' if all(s['passed'] for s in scores) else 'FAILED_TASK_QUALITY',
                independent_paired_starts=1,GPU_pair=config[0]['gpus'],
                metrics_guard0={f:metrics[0][f] for f in fields},metrics_guard1={f:metrics[1][f] for f in fields},
                guard1_over_guard0={f:metrics[1][f]/metrics[0][f] if metrics[0][f]>0 else None for f in fields},
                chunk_means_guard0=chunk[0]['means'],chunk_means_guard1=chunk[1]['means'],
                mean_host_boundaries=boundaries(names[1],names[0]),quality=scores)
            cells.append(row)
    result=dict(time=time.time(),cells=cells,evidence='MEASURED_ONLINE_WITH_TRACE_ANALYSIS',
        all_complete=all(r['status']=='MEASURED_ONLINE' for r in cells),
        scope='Common pre-allocation policy intervention. Preserve every improvement and regression. '
            'Each within-axis A/B uses the same source,pair,requests,arrival and natural output contract. '
            'Actual BI0 EOS counts can differ. n1 has no independent-start confidence claim. '
            'Host chunk gaps and stage boundaries are not GPU-exclusive time. '
            'A guard benefit need not be an intrinsic layer-GEMM benefit. '
            'The QPS2.8 whole block moved before any original job started. The migration receipt preserves that fact.')
    write(ROOT/'analysis/completion_guard_common_policy_intervention_v1.json',result)
    print([(r['QPS'],r['variant'],r['status']) for r in cells],flush=True)


if __name__=='__main__':
    main()
