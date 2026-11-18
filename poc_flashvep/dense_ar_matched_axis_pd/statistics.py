"""Use independent starts for uncertainty. Keep discovery separate."""
import collections
import csv
import json
import statistics
import time
from .common import ROOT,write
from .analysis import metrics
from ..dense_ar_layer_token_pd.statistics_report import mean_interval,ratio_interval

KEYS=('mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms',
      'E2E_p95_ms','maximum_ITL_ms','output_tokens_s','completed_QPS')


def analysis_stage(stage):
    if stage.startswith(('batch1_', 'cohort2_')) or stage in ('confirmation', 'admission_control'):
        return stage
    return 'discovery'


def collect():
    rows=[]
    inventory=[]
    superseded=ROOT/'state/superseded_prelaunch_v1_matrix.json'
    superseded_names=set(json.loads(superseded.read_text())['names']) if superseded.exists() else set()
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or cfg.get('name')!=path.stem:
            continue
        folder=ROOT/'raw'/cfg['name']
        receipt_path=folder/'dispatcher_result.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        status='NOT_RUN' if receipt is None else 'FAILED' if not receipt.get('complete') else 'CORRECTNESS_ONLY' if cfg.get('correctness_only') else 'MEASURED'
        inventory.append(dict(name=cfg['name'],status=status,method=cfg['evaluation_arm'],
            cap=cfg['cap'],QPS=cfg['arrival_rate'],layers=cfg['static_layers'],
            count=cfg['count'],phase=cfg['phase'],receipt=receipt,
            superseded_before_clean_performance=cfg['name'] in superseded_names,
            execution_contract_version=cfg.get('execution_contract_version',2)))
        if status!='MEASURED' or cfg.get('auxiliary_only'):
            continue
        if cfg.get('matrix_stage') not in ('discovery','motivation_odd','confirmation','admission_control',
                                        'batch1_discovery','batch1_confirmation',
                                        'cohort2_discovery','cohort2_confirmation'):
            continue
        if cfg.get('batch_matched_control'):
            suffix='_fixed_cohort_gate.json' if cfg.get('matched_D_cohort_size') else '_matched_batch_gate.json'
            checked=ROOT/'analysis'/(cfg['name']+suffix)
            if not checked.exists():
                continue
            assert json.loads(checked.read_text())['passed'],cfg['name']
        if not (folder/'unified_worker_gate.json').exists():
            continue
        result_path=ROOT/'analysis'/(cfg['name']+'_metrics.json')
        value=json.loads(result_path.read_text()) if result_path.exists() else metrics(cfg['name'])
        if ('Common_Balanced_OutputSweep' not in value['SLO_regimes'] or
                cfg['arm']=='PD_G1' and ('per_request_P_query_coverage' not in value or
                value.get('analysis_schema_version')!=3)):
            value=metrics(cfg['name'])
        assert value['output_gate']['passed'] and value['unified_worker_gate']
        row=dict(name=cfg['name'],method=cfg['evaluation_arm'],
            layers=cfg['static_layers'] if cfg['evaluation_arm'] in ('Layer','Token') else 0,
            cap=cfg['cap'],QPS=cfg['arrival_rate'],seed=cfg['arrival_seed'],
            count=cfg['count'],dataset=cfg['dataset'],admission=cfg['D_admission_batch'],
            stage=cfg['matrix_stage'],regimes=value['SLO_regimes'],
            actual_shapes=value['batch_geometry_counts'],
            source_groups=value['source_groups'],destination_groups=value['destination_groups'],
            work=value['work_accounting'],actual_query_coverage=value.get('actual_query_coverage_gate'),
            exact_GEMM_row_gate=value.get('exact_GEMM_row_gate'),
            native_observations=value['native_observations'],
            queue_observations=value.get('coordinator_queue_observations'),
            actual_arrival_QPS=value['realized_arrival_QPS'],
            scheduled_trace_QPS=value['scheduled_trace_QPS'],
            evidence='DIAGNOSTIC' if cfg.get('execution_events') or cfg.get('nsys_bounded_capture') else 'MEASURED')
        row.update({key:value[key] for key in KEYS})
        rows.append(row)
    return rows,inventory


def main():
    rows,inventory=collect()
    groups=collections.defaultdict(list)
    for row in rows:
        phase=analysis_stage(row['stage'])
        groups[(phase,row['method'],row['layers'],row['cap'],row['QPS'],row['count'],row['admission'])].append(row)
    aggregate=[]
    for (phase,method,layers,cap,qps,count,admission),items in sorted(groups.items()):
        assert len({r['seed'] for r in items})==len(items),'Do not count repeated same-seed reruns as independent arrivals'
        aggregate.append(dict(stage=phase,method=method,layers=layers,cap=cap,QPS=qps,
            count=count,admission=admission,starts=len(items),seeds=[r['seed'] for r in items],
            names=[r['name'] for r in items],
            metrics={key:dict(mean=statistics.mean(r[key] for r in items),
                CI95=mean_interval([r[key] for r in items])) for key in KEYS},
            regimes={regime:{key:dict(mean=statistics.mean(r['regimes'][regime][key] for r in items),
                CI95=mean_interval([r['regimes'][regime][key] for r in items]))
                for key in ('TTFT_attainment','Decode_attainment','Joint_attainment',
                            'Output_token_SLO_goodput','Request_SLO_goodput')}
                for regime in ('TTFT_priority','Balanced','Decode_priority','Common_Balanced_OutputSweep')},
            evidence='MEASURED',scope='Per-start metrics. Independent-start pointwise bootstrap. No CI at n1. Finite-cohort drain rates are not sustainable capacity.'))
    comparisons=[]
    conditions=sorted({(stage,cap,qps,count,admission) for stage,_,_,cap,qps,count,admission in groups})
    for stage,cap,qps,count,admission in conditions:
        for layers in (4,8,12):
            for ref_method,candidate in (('Token','Layer'),('Native','Layer'),('Native','Token')):
                a=groups.get((stage,ref_method,layers if ref_method=='Token' else 0,cap,qps,count,admission),[])
                b=groups.get((stage,candidate,layers,cap,qps,count,admission),[])
                aa={r['seed']:r for r in a};bb={r['seed']:r for r in b};seeds=sorted(set(aa)&set(bb))
                if not seeds:
                    continue
                for seed in seeds:
                    ca=json.loads((ROOT/'configs'/(aa[seed]['name']+'.json')).read_text())
                    cb=json.loads((ROOT/'configs'/(bb[seed]['name']+'.json')).read_text())
                    assert all(ca[k]==cb[k] for k in ('arrival_offsets','request_caps','dataset','gpus',
                        'token_budget','D_token_budget','P_slots','D_slots','D_admission_batch'))
                tests={key:ratio_interval([aa[s][key] for s in seeds],[bb[s][key] for s in seeds]) for key in KEYS}
                regime_tests={regime:{key:ratio_interval(
                    [aa[s]['regimes'][regime][key] for s in seeds],
                    [bb[s]['regimes'][regime][key] for s in seeds])
                    for key in ('Joint_attainment','Output_token_SLO_goodput')}
                    for regime in ('Balanced','Decode_priority','TTFT_priority','Common_Balanced_OutputSweep')}
                upper=[tests[k]['ratio_CI95'] for k in ('ITL_p99_ms','TPOT_p95_ms')]
                safety=('PASS' if len(seeds)>=3 and all(c is not None and c[1]<=1.05 for c in upper) else 'NOT_ESTABLISHED') if ref_method=='Native' else 'NOT_APPLICABLE'
                comparisons.append(dict(stage=stage,reference=ref_method,candidate=candidate,
                    layers=layers,cap=cap,QPS=qps,count=count,admission=admission,
                    starts=len(seeds),seeds=seeds,metrics=tests,regimes=regime_tests,
                    Native_relative_5pct=safety,
                    scope='Paired independent starts at identical prompt pool, count, output, arrival seed and admission cap. Discovery and confirmation are not pooled.'))
    write(ROOT/'analysis/experiment_inventory.json',dict(time=time.time(),
        counts=dict(collections.Counter(r['status'] for r in inventory)),runs=inventory))
    write(ROOT/'analysis/independent_start_statistics.json',dict(time=time.time(),
        starts=rows,aggregate=aggregate,comparisons=comparisons,
        scope='95% pointwise paired-start bootstrap. No multiple-comparison correction. No request/token resampling as independent starts.'))
    with (ROOT/'analysis/per_start_metrics.csv').open('w',newline='') as stream:
        fields=['name','stage','method','layers','cap','QPS','count','admission','seed',*KEYS,
                'Balanced_Joint','Balanced_SLO_tokens_s','Decode_Joint','Decode_SLO_tokens_s']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            values={key:row[key] for key in fields if key in row}
            values.update(Balanced_Joint=row['regimes']['Balanced']['Joint_attainment'],
                Balanced_SLO_tokens_s=row['regimes']['Balanced']['Output_token_SLO_goodput'],
                Decode_Joint=row['regimes']['Decode_priority']['Joint_attainment'],
                Decode_SLO_tokens_s=row['regimes']['Decode_priority']['Output_token_SLO_goodput'])
            writer.writerow(values)
    print('STATISTICS',len(rows),len(comparisons),flush=True)


if __name__=='__main__':
    main()
