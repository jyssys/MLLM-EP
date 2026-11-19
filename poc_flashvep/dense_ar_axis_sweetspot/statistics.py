"""Resample paired server starts, never requests or individual tokens."""
import collections
import json
import statistics
import time
from .common import ROOT,write
from .resampling import mean_interval,ratio_interval,METHOD

KEYS=('mean_TTFT_ms','TTFT_p95_ms','mean_TBT_ms','ITL_p99_ms','TPOT_p95_ms',
      'maximum_ITL_ms','output_tokens_s','completed_QPS')


def collect():
    starts=[];inventory=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or cfg.get('name')!=path.stem:
            continue
        folder=ROOT/'raw'/cfg['name']
        receipt=folder/'dispatcher_result.json'
        raw=json.loads(receipt.read_text()) if receipt.exists() else None
        status=('NOT_RUN' if raw is None else 'FAILED' if not raw['complete'] else
            'CORRECTNESS_ONLY' if cfg.get('correctness_only') else
            'ISOLATED_KERNEL_MEASUREMENT' if cfg.get('auxiliary_only') else
            'DIAGNOSTIC' if cfg.get('execution_events') else 'MEASURED_ONLINE')
        inventory.append(dict(name=cfg['name'],status=status,stage=cfg.get('matrix_stage'),
            method=cfg['evaluation_arm'],cut=cfg['static_layers'],output=cfg['cap'],
            QPS=cfg['arrival_rate'],seed=cfg['arrival_seed'],count=cfg['count'],
            batch=cfg['D_admission_batch'],source_window=cfg.get('source_outstanding_limit'),receipt=raw))
        target=ROOT/'analysis'/(cfg['name']+'_service_queue_audit.json')
        if status!='MEASURED_ONLINE' or not target.exists() or cfg.get('matrix_stage','').startswith('event_observer_'):
            continue
        row=json.loads(target.read_text())
        row.update(stage=cfg['matrix_stage'],dataset=cfg['dataset'],arrival_offsets=cfg['arrival_offsets'])
        starts.append(row)
    return starts,inventory


def main():
    starts,inventory=collect()
    groups=collections.defaultdict(list)
    for r in starts:
        groups[r['stage'],r['method'],r['layers'],r['output'],r['QPS'],r['count'],r['actual_batch'],r['source_window']].append(r)
    aggregate=[]
    for key,rows in sorted(groups.items()):
        stage,a,m,o,q,n,b,w=key
        assert len({r['seed'] for r in rows})==len(rows)
        aggregate.append(dict(stage=stage,method=a,layers=m,output=o,QPS=q,count=n,batch=b,window=w,
            starts=len(rows),seeds=[r['seed'] for r in rows],names=[r['name'] for r in rows],
            metrics={k:dict(mean=statistics.mean(r['metric_summary'][k] for r in rows),
                CI95=mean_interval([r['metric_summary'][k] for r in rows])) for k in KEYS},
            common_Balanced={k:dict(mean=statistics.mean(r['common_Balanced'][k] for r in rows),
                CI95=mean_interval([r['common_Balanced'][k] for r in rows]))
                for k in ('Joint_attainment','TTFT_attainment','Decode_attainment','Output_token_SLO_goodput')},
            host_boundaries_ms={k:statistics.mean(r[k] for r in rows) for k in
                ('P_submit_to_ready_mean_ms','D_host_slot_per_request_ms','Ready_D_mean_ms','P_external_wait_mean_ms')},
            evidence='MEASURED_ONLINE',scope='Finite-cohort client metrics. No CI at one start. No sustainable-capacity inference.'))
    comparisons=[]
    conditions={tuple(k[i] for i in (0,3,4,5,6,7)) for k in groups}
    for stage,o,q,n,b,w in sorted(conditions):
        for m in (2,4,8,12):
            for ref,candidate in (('Token','Layer'),('Native','Layer'),('Native','Token')):
                a=groups.get((stage,ref,m if ref=='Token' else 0,o,q,n,b,w),[])
                c=groups.get((stage,candidate,m,o,q,n,b,w),[])
                aa={r['seed']:r for r in a};cc={r['seed']:r for r in c}
                seeds=sorted(set(aa)&set(cc))
                if not seeds:
                    continue
                assert all(aa[s]['arrival_offsets']==cc[s]['arrival_offsets'] and aa[s]['dataset']==cc[s]['dataset'] for s in seeds)
                tests={k:ratio_interval([aa[s]['metric_summary'][k] for s in seeds],
                    [cc[s]['metric_summary'][k] for s in seeds]) for k in KEYS}
                goodput=ratio_interval([aa[s]['common_Balanced']['Output_token_SLO_goodput'] for s in seeds],
                    [cc[s]['common_Balanced']['Output_token_SLO_goodput'] for s in seeds])
                safety='NOT_APPLICABLE'
                if ref=='Native':
                    safety='PASS' if len(seeds)>=3 and all(tests[k]['ratio_CI95'] is not None
                        and tests[k]['ratio_CI95'][1]<=1.05 for k in ('ITL_p99_ms','TPOT_p95_ms')) else 'NOT_ESTABLISHED'
                comparisons.append(dict(stage=stage,reference=ref,candidate=candidate,layers=m,output=o,QPS=q,
                    count=n,batch=b,window=w,starts=len(seeds),seeds=seeds,metrics=tests,
                    common_Balanced_goodput=goodput,Native_relative_5pct=safety,
                    scope='Candidate/reference ratio, paired by independent server start and arrival seed. Pointwise95% bootstrap. No multiplicity adjustment.'))
    write(ROOT/'analysis/experiment_inventory.json',dict(time=time.time(),
        counts=dict(collections.Counter(r['status'] for r in inventory)),runs=inventory))
    write(ROOT/'analysis/independent_start_statistics.json',dict(time=time.time(),starts=starts,
        aggregate=aggregate,comparisons=comparisons,CI_method=METHOD,
        scope='New campaign only. Frozen historical results remain separate. Same-seed observer runs are not independent confirmation.'))
    print('NEW_STATISTICS',len(starts),len(comparisons),flush=True)


if __name__=='__main__':
    main()
