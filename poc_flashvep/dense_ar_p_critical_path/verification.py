"""Check raw metrics,immutable contracts and real process state."""
import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import time
from .common import ROOT,SOURCE,REPO,PYTHON,MODEL_REVISION,write,digest,bind,preserve


def cpu_regression():
    preserve();rows=[]
    commands=[['-m','unittest','poc_flashvep.dense_ar_p_critical_path.test_protocol'],
        ['-m','poc_flashvep.dense_ar_p_critical_path.critical_analysis','test'],
        ['-m','poc_flashvep.dense_ar_p_critical_path.kernel_analysis'],
        ['-m','poc_flashvep.dense_ar_p_critical_path.coordinator_analysis','test']]
    for suffix in commands:
        command=[str(PYTHON),'-B']+suffix
        result=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
        rows.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    files=[]
    for path in sorted(SOURCE.glob('*.py')):
        compile(path.read_bytes(),str(path),'exec');files.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
    passed=all(row['exit_code']==0 for row in rows)
    value=dict(time=time.time(),passed=passed,commands=rows,source_files=files,
        scope='CPU-only tests cover the installed observer seam,source retirement fences and ownership,idempotence,GC restoration,buffered I/O, '
            'distributed strict-state schema,stable-source corruption rejection,timeline union,category overlap and delivery interval identity. '
            'Syntax compilation does not create a GPU context or qualify GPU correctness. Source hashes describe this audit point.')
    write(ROOT/'analysis/cpu_regression_audit.json',value);preserve();assert passed,rows
    print('CPU_REGRESSION_PASS',len(files),'files',len(rows),'commands',flush=True)
    return value


def live_command(folder):
    path=folder/'command.json'
    if not path.exists():return None
    row=json.loads(path.read_text());pid=row['pid'];proc=Path('/proc',str(pid))
    try:
        words=(proc/'cmdline').read_bytes().decode().split('\0')
        expected=str(ROOT/'configs'/(folder.name+'.json'))
        if expected not in words:return None
        assert proc.stat().st_uid==os.getuid()
        return dict(pid=pid,words=words)
    except (FileNotFoundError,ProcessLookupError):return None


def inventory(final=False):
    runner=bind();rows=[];live=[];interrupted=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or not all(key in cfg for key in ('name','dataset','gpus','batch_invariant')):continue
        name=cfg['name'];assert name==path.stem
        assert cfg['batch_invariant'] is False and cfg['token_budget']==8192 and cfg['max_num_seqs']==1024
        assert digest(ROOT/'data'/cfg['dataset'])==cfg['dataset_SHA256'],name
        folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json';command=live_command(folder) if folder.exists() else None
        if receipt.exists():status='COMPLETE' if json.loads(receipt.read_text())['complete'] else 'FAILED'
        elif command:status='RUNNING';live.append(dict(name=name,**command))
        elif folder.exists():status='INTERRUPTED_NO_LIVE_HANDLE';interrupted.append(name)
        else:status='NOT_RUN'
        proof=None
        if status=='COMPLETE' and not cfg.get('correctness_only'):
            gate=ROOT/'state'/cfg['qualification_gate_v2'];proof=json.loads(gate.read_text())
            assert proof['passed'] and proof['signature']==runner.signature(cfg),name
            assert proof['source_capsule_SHA256']==cfg['source_capsule_SHA256'],name
        row=dict(name=name,status=status,dataset_SHA256=cfg['dataset_SHA256'],config_SHA256=digest(path),
            source_archive_SHA256=cfg['source_capsule_SHA256'],GPU_pair=cfg['gpus'],QPS=cfg.get('qps'),seed=cfg.get('seed'),
            actual_batch_status='RAW_RUNTIME_GEOMETRY' if status=='COMPLETE' else 'NOT_MEASURED',
            B=8192,S=1024,KV_block_tokens=cfg.get('KV_block_tokens',cfg.get('block_size')),
            source_credit=cfg.get('continuous_source_credit_limit',0),D_cap=cfg.get('D_running_admission_cap'),
            correctness_only=cfg.get('correctness_only',False),qualification_gate=cfg.get('qualification_gate_v2'),
            source_credit_release_on_retirement=cfg.get('source_credit_release_on_retirement',False),
            buffer_coordinator_evidence=cfg.get('buffer_coordinator_evidence',False),
            defer_coordinator_cyclic_gc=cfg.get('defer_coordinator_cyclic_gc',False),
            stable_worker_source_path=cfg.get('stable_worker_source_path',False))
        if folder.exists() and (folder/'input_provenance.json').exists():
            provenance=json.loads((folder/'input_provenance.json').read_text())
            assert provenance['model_revision']==MODEL_REVISION and provenance['dataset_SHA256']==cfg['dataset_SHA256']
        runtime=[]
        if folder.exists():
            for runtime_path in sorted(folder.glob('actual_runtime_*.json')):
                actual=json.loads(runtime_path.read_text())
                assert actual['scheduler_B']==8192 and actual['scheduler_S']==1024,(name,runtime_path.name)
                assert actual['batch_invariant'] is False,(name,runtime_path.name)
                runtime.append(dict(file=runtime_path.name,SHA256=digest(runtime_path),role=actual.get('role'),
                    GPU_UUID=actual.get('uuid'),B=actual['scheduler_B'],S=actual['scheduler_S'],
                    KV_block_tokens=actual.get('KV_block_tokens'),KV_bytes=actual.get('KV_bytes'),
                    compile_mode=actual.get('compile_mode'),cudagraph_mode=actual.get('cudagraph_mode'),
                    chunked_prefill=actual.get('chunked_prefill'),D_cap=actual.get('D_running_admission_cap'),
                    parameter_dtypes=actual.get('parameter_dtypes')))
        row['actual_worker_configuration']=runtime
        rows.append(row)
    value=dict(time=time.time(),passed=not interrupted and (not final or not live),final=final,rows=rows,live=live,interrupted=interrupted,
        counts={status:sum(row['status']==status for row in rows) for status in sorted({row['status'] for row in rows})},
        scope='Enumerate every immutable runtime configuration,including failures and NOT_RUN. '
            'A live raw folder requires a matching owned process command,not only a lock file. '
            'Every completed timing run retains its exact composition proof,source,dataset,B/S and model revision. '
            'A complete dispatcher receipt is not itself an accuracy,decode-safety or observer-overhead pass.')
    write(ROOT/'analysis/configuration_and_gate_inventory.json',value)
    from .critical_analysis import csv_save
    csv_save(ROOT/'figures/configuration_and_gate_inventory.csv',rows)
    print('CONFIGURATION_INVENTORY',value['counts'],'live',len(live),'interrupted',len(interrupted),flush=True)
    if final:assert value['passed'],value
    return value


def metrics(final=False):
    bind()
    from ..dense_ar_longbench400_pd.independent_metric_audit import one
    results=[];failed=[];missing=[];pending=[];errors=[];non_client=[];new_caches=[]
    for folder in sorted((ROOT/'raw').iterdir()):
        if not folder.is_dir():continue
        receipt=folder/'dispatcher_result.json'
        if not receipt.exists():
            pending.append(dict(name=folder.name,live=live_command(folder)));continue
        if not json.loads(receipt.read_text())['complete']:failed.append(folder.name);continue
        if not (folder/'window.json').exists() or not (folder/'requests.jsonl').exists():
            non_client.append(folder.name);continue
        cached=ROOT/'analysis'/(folder.name+'_metrics.json')
        if not cached.exists():
            try:
                from ..dense_ar_output_aware_v2.derived_metrics import summarize
                summarize(folder.name);new_caches.append(folder.name)
            except Exception as error:
                missing.append(dict(name=folder.name,error=repr(error)));continue
        try:results.append(one(folder.name))
        except Exception as error:errors.append(dict(name=folder.name,error=repr(error)))
    value=dict(time=time.time(),passed=not errors and not missing and (not final or not pending),final=final,
        results=results,preserved_failed_dispatches=failed,not_client_timing_records=non_client,
        missing_cached_metrics=missing,reconstructed_missing_caches=new_caches,pending=pending,errors=errors,evidence='RAW_CLIENT_METRIC_RECONSTRUCTION',
        scope='Read immutable request events and cohort windows. Reconstruct25 client metrics independently of the primary metric implementation. '
            'Check token counts,co-delivered zero ITLs,completion boundary,frozen SLO and exact forced replay lengths. '
            'Failed dispatches retain failure evidence. Correctness-only teachers without client windows have no timing claim. '
            'A partial audit is not a final no-live-job audit. Accuracy and profile interference need separate gates.')
    write(ROOT/'analysis/raw_metric_recalculation.json',value)
    print('RAW_METRIC_AUDIT',len(results),'verified',len(errors),'errors',len(missing),'missing',len(pending),'pending',flush=True)
    if final:assert value['passed'],value
    return value


def interval_union_seconds(intervals):
    total=0.;end=None
    for begin,finish in sorted(intervals):
        assert finish>=begin
        if end is None or begin>=end:total+=finish-begin
        elif finish>end:total+=finish-end
        end=finish if end is None else max(end,finish)
    return total


def budget(final=False):
    from .common import START
    preserve();rows=[];pending=[];intervals={gpu:[] for gpu in range(8)}
    for family,parent in (('Serving',ROOT/'raw'),('Attention PoC',ROOT/'attention_poc/raw')):
        for folder in sorted(parent.iterdir()):
            if not folder.is_dir() or not (folder/'command.json').exists():continue
            command=json.loads((folder/'command.json').read_text());path=folder/'dispatcher_result.json'
            if not path.exists():pending.append(dict(family=family,name=folder.name));continue
            receipt=json.loads(path.read_text());duration=receipt['elapsed_seconds'];start=command['start_epoch']
            assert duration>=0 and set(command['GPUs'])<=set(range(8))
            for gpu in command['GPUs']:intervals[gpu].append((start,start+duration))
            rows.append(dict(family=family,name=folder.name,GPUs=command['GPUs'],start_epoch=start,
                elapsed_seconds=duration,reserved_GPU_process_hours=len(command['GPUs'])*duration/3600,
                dispatcher_complete=receipt['complete'],command_SHA256=digest(folder/'command.json'),
                receipt_SHA256=digest(path)))
    frozen=json.loads((ROOT/'state/budget.json').read_text())
    latest=max((row['start_epoch']+row['elapsed_seconds'] for row in rows),default=START)
    per_gpu={str(gpu):interval_union_seconds(values)/3600 for gpu,values in intervals.items()}
    passed=not pending and latest<=frozen['deadline_epoch']
    value=dict(time=time.time(),passed=passed,final=final,rows=rows,pending=pending,
        approved_maximum_wall_hours=frozen['maximum_hours'],approved_GPU_count=8,
        observed_dispatch_span_hours=(latest-START)/3600,
        sum_reserved_GPU_process_hours=sum(row['reserved_GPU_process_hours'] for row in rows),
        reserved_dispatch_interval_union_GPU_hours=sum(per_gpu.values()),per_GPU_union_hours=per_gpu,
        scope='Elapsed dispatcher-process intervals multiplied by their reserved GPU set. Include failed startup and correctness controls. '
            'Union intervals per GPU to avoid double counting. This is not GPU-exclusive kernel time or GPU utilization. '
            'The attention capture reserves two devices even when only one computes. Burner time and CPU-only analysis are excluded. '
            'Do not treat this accounting as a serving-performance sample count.')
    write(ROOT/'analysis/gpu_budget_accounting.json',value)
    from .critical_analysis import csv_save
    csv_save(ROOT/'figures/gpu_budget_dispatches.csv',rows)
    print('GPU_DISPATCH_BUDGET',len(rows),value['reserved_dispatch_interval_union_GPU_hours'],'GPU-hours','pending',len(pending),flush=True)
    if final:assert passed,value
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('cpu','inventory','metrics','budget'))
    parser.add_argument('--final',action='store_true');args=parser.parse_args()
    if args.action=='cpu':cpu_regression()
    else:globals()[args.action](args.final)
