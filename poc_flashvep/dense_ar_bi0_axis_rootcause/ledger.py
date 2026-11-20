"""Export every launch, failure and requested matrix cell without imputation."""
import collections
import csv
import json
from .common import ROOT,write,digest


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default


def catalogue():
    known={}
    lease=read(ROOT/'state/GPU_lease_current.json',{})
    for path in sorted((ROOT/'analysis').glob('*.json')):
        value=read(path,{})
        if not isinstance(value,dict):continue
        for row in value.get('results',[]):
            if isinstance(row,dict) and isinstance(row.get('name'),str) and row.get('status'):
                known.setdefault(row['name'],[]).append(dict(status=row['status'],file=path.name,
                    task_quality_CI_passed=row.get('quality',{}).get('CI_passed')))
    profile=read(ROOT/'analysis/P_kernel_profile_qualification_summary.json',{}).get('results',[])
    profile += read(ROOT/'analysis/D_suffix_profile_qualification_summary.json',{}).get('results',[])
    profile += read(ROOT/'analysis/D_kernelonly_suffix_profile_qualification_summary.json',{}).get('results',[])
    qualified={r['names']['profile'] for r in profile if r.get('qualified')}
    rows=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=read(path,{})
        if not isinstance(cfg,dict) or not cfg.get('entry_module') or not cfg.get('name'):continue
        name=cfg['name'];folder=ROOT/'raw'/name;receipt=read(folder/'dispatcher_result.json',{})
        live=read(folder/'status.json',{});metric=read(ROOT/'analysis'/(name+'_metrics.json'),{})
        if receipt:
            status='FAILED' if not receipt.get('complete') else 'CORRECTNESS_ONLY' if cfg.get('correctness_only') else 'MEASURED_ONLINE'
        elif lease.get('status')=='RUNNING' and lease.get('name')==name:status='RUNNING'
        else:status='FAILED' if (folder/'command.json').exists() else 'NOT_RUN'
        annotations=known.get(name,[])
        if receipt.get('complete'):
            observed={r['status'] for r in annotations}
            if 'DIAGNOSTIC' in observed:status='DIAGNOSTIC'
            elif 'MEASURED_ONLINE' in observed:status='MEASURED_ONLINE'
        is_profile=bool(cfg.get('correlated_profile') or cfg.get('kernel_activity'))
        scope='DIAGNOSTIC' if cfg.get('diagnostic_controlled_output_replay') or is_profile else 'NATURAL_EOS'
        quality=next((r['task_quality_CI_passed'] for r in annotations if r['task_quality_CI_passed'] is not None),None)
        evidence_level=status if status in ('FAILED','NOT_RUN','RUNNING') else 'DIAGNOSTIC' if (is_profile and name not in qualified) or quality is False else status
        runtime={role:read(folder/f'actual_runtime_{role}.json') for role in ('prefill','decode','single','replica0','replica1')}
        runtime={k:v for k,v in runtime.items() if v is not None}
        geometry=read(ROOT/'analysis'/(name+'_timeline_analysis.json'),{})
        timeline=read(ROOT/'analysis'/(name+'_axis_timeline.json'),{})
        row=dict(name=name,status=status,evidence_level=evidence_level,
            execution_complete=receipt.get('complete',False),measurement_scope=scope,result_annotations=annotations,
            execution_returncode=receipt.get('returncode'),worker_exit_status=read(folder/'worker_exits.json'),
            method=cfg.get('method'),axis=cfg.get('axis_kind'),cut=cfg.get('axis_cut',0),
            N_requested=cfg.get('count'),N_completed=metric.get('n',live.get('completed')),
            offered_QPS=cfg.get('offered_QPS'),arrival_seed=cfg.get('arrival_seed'),BI=int(bool(cfg.get('batch_invariant'))),
            B=cfg.get('token_budget'),S=cfg.get('max_num_seqs'),credit=cfg.get('source_credit_limit'),
            hidden_source_slots=cfg.get('hidden_source_slots'),hidden_destination_slots=cfg.get('hidden_destination_slots',1),
            KV_block=cfg.get('KV_block_tokens'),KV_bytes_per_GPU=42*1024**3,
            controlled_pipeline=cfg.get('controlled_pipeline',False),execution_family=cfg.get('execution_family'),
            dataset=cfg.get('dataset'),dataset_SHA256=cfg.get('dataset_SHA256'),
            model_revision='b968826d9c46dd6066d109eabc6255188de91218',
            source_archive_SHA256=receipt.get('source_SHA256',read(folder/'command.json',{}).get('source_archive_SHA256')),
            configuration_SHA256=digest(path),runtime=runtime,geometry=geometry.get('geometry'),
            task_quality_CI_passed=quality,
            all_prompt_work_accounted=timeline.get('all_prompt_work_accounted'),
            lifetime={p.name:read(p).get('passed') for p in folder.glob('*lifetime*.json')},
            profile_observer_and_attribution_qualified=name in qualified if is_profile else None,
            metrics={k:v for k,v in metric.items() if k not in ('per_request','by_task')},
            raw_directory=str(folder.relative_to(ROOT)),failure=live.get('error') or ('Dispatcher or worker exit gate failed. See preserved logs.' if status=='FAILED' else None),
            dispatch_scope='Prepared source/config only. No command or completed dispatch receipt.' if status=='NOT_RUN' and folder.exists() else None,
            finite_cohort=True,sustainable_capacity=False)
        rows.append(row)
    return rows


def matrices():
    pilot=read(ROOT/'analysis/pilot50_all_cuts_qps_v1.json',{}).get('results',[])
    full=read(ROOT/'analysis/last12_confirmation_v1.json',{}).get('results',[])
    repair=read(ROOT/'analysis/last12_native_reference_repair.json')
    if repair:full=[r for r in full if r['name']!=repair['original']]+[repair['replacement']]
    full += read(ROOT/'analysis/full200_allcuts_confirmation_v1.json',{}).get('results',[])
    rows=[]
    for N,values in ((50,pilot),(200,full)):
        for q in (.4,.7,1.,1.2,1.65,2.,2.35):
            methods=[('Native',0)]+[(axis,cut) for cut in (2,4,8,12) for axis in ('Layer','Token')]
            for axis,cut in methods:
                matches=[r for r in values if r.get('axis')==axis and r.get('cut')==cut and r.get('QPS')==q]
                measured=[r for r in matches if r.get('status')=='MEASURED_ONLINE']
                qualified=bool(measured) and all(r.get('quality',{}).get('CI_passed',False) for r in measured)
                rows.append(dict(N=N,axis=axis,cut=cut,offered_QPS=q,
                    evidence='MEASURED_ONLINE' if qualified else 'DIAGNOSTIC' if measured else 'NOT_RUN',
                    quality_CI_passed=qualified if measured else None,
                    independent_starts=len(measured),arrival_seeds=[r['seed'] for r in measured],
                    names=[r['name'] for r in measured],
                    reason=None if qualified else 'Frozen task-quality CI gate not passed.' if measured else 'No completed measurement in this family. N50 never fills a Full200 cell.'))
    return rows


def main():
    rows=catalogue();matrix=matrices()
    write(ROOT/'analysis/complete_execution_ledger.json',dict(runs=rows,
        counts=dict(collections.Counter(r['status'] for r in rows)),
        scope='Every new launch configuration, including failed and correctness-only jobs. Producer status COMPLETE cannot override a failed worker/dispatcher receipt.'))
    fields=['name','status','evidence_level','execution_complete','execution_returncode','measurement_scope','method','axis','cut','N_requested','N_completed','offered_QPS','arrival_seed',
        'BI','B','S','credit','hidden_source_slots','hidden_destination_slots','KV_block','controlled_pipeline','execution_family',
        'dataset_SHA256','model_revision','source_archive_SHA256','configuration_SHA256','all_prompt_work_accounted','task_quality_CI_passed',
        'TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms','SLO_qualified_output_tokens_per_second','generated_output_tokens_per_second','output_tokens','failure']
    with (ROOT/'analysis/complete_execution_ledger.csv').open('w',newline='') as stream:
        out=csv.DictWriter(stream,fieldnames=fields);out.writeheader()
        for row in rows:
            values=dict(row,**row['metrics']);out.writerow({key:values.get(key) for key in fields})
    write(ROOT/'analysis/requested_N50_and_full200_matrix.json',dict(rows=matrix,
        scope='Keep Discovery N50, Main Full200 and independent-start counts separate. No interpolation or request-level pseudo-replication.'))
    print(json.dumps(dict(runs=len(rows),counts=dict(collections.Counter(r['status'] for r in rows)))))


if __name__=='__main__':main()
