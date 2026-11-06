"""Keep execution, state qualification, and expansion decisions separate."""
import csv
import time
from .common import *
from .analyze import table


def main():
    import subprocess
    command=['/home/esjung/.venvs/looped-pd-abc/bin/python','-m','pytest','--import-mode=importlib','-q',
        'poc_flashvep/d_local_cross_depth_batching/test_contract.py','poc_flashvep/fluidtoken_loop_split/test_probe_quantum.py']
    checked=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
    write(ROOT/'cpu_regression_final.json',dict(command=command,returncode=checked.returncode,
        stdout=checked.stdout,stderr=checked.stderr,time=time.time(),
        scope='CPU contracts only. GPU output, KV, hidden state, and lifetime qualification remain separate.'))
    assert checked.returncode==0,checked.stdout+checked.stderr
    from .qualification import compare_outputs
    for model in ('ouro','huginn'):
        name=f'serve_{model}_paced_heavy_fixed25_'+('clean_restart_v2' if model=='ouro' else 'v1')
        path=ROOT/'raw'/name/'dispatcher_result.json'
        if path.exists() and json.loads(path.read_text()).get('complete'):
            compare_outputs(f'local_{model}_native_n128_s2026110701_off',name,128)
    matrix=[];failures=[];decisions={}
    commands=ROOT/'commands.jsonl'
    launched={json.loads(line)['name'] for line in commands.read_text().splitlines()} if commands.exists() else set()
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if 'name' not in cfg:continue
        folder=ROOT/'raw'/cfg['name'];receipt=folder/'dispatcher_result.json'
        result=json.loads(receipt.read_text()) if receipt.exists() else {}
        status=result.get('status') or ('COMPLETE' if result.get('complete') else 'FAILED' if result else 'NOT_RUN_SUPERSEDED_OR_BUDGET_PRIORITY')
        skip=ROOT/'state'/f'recovery_skip_{cfg["name"]}.json'
        if not result and skip.exists():status=json.loads(skip.read_text())['status']
        if result and not result.get('complete') and status=='RUNNING':status='FAILED_TERMINATED'
        matrix.append(dict(name=cfg['name'],model=cfg['model'],method=cfg.get('dlocal_method'),worker=cfg['worker'],
            count=cfg['count'],status=status,complete=result.get('complete',False),returncode=result.get('returncode'),
            guarded_worker_started=cfg['name'] in launched,
            timing_qualification=result.get('timing_qualification','NORMAL_GUARDED_EXECUTION' if result else 'NOT_RUN'),
            config=str(path.relative_to(ROOT)),log=f'logs/{cfg["name"]}.log'))
        if result and not result.get('complete'):
            failures.append(dict(name=cfg['name'],kind='EXECUTION_FAILURE',status=status,
                log=f'logs/{cfg["name"]}.log',receipt=str(receipt.relative_to(ROOT))))
        q=folder/'local_qualification.json'
        if q.exists() and not json.loads(q.read_text())['passed']:
            failures.append(dict(name=cfg['name'],kind='BLOCKED_CORRECTNESS',status='OUTPUT_OR_VALID_STATE_MISMATCH',
                log=f'logs/{cfg["name"]}.log',receipt=str(q.relative_to(ROOT))))
    for file,gate in (('instrumentation_qualification.csv','qualified'),('service_marker_qualification.csv','qualified')):
        path=ROOT/file
        if not path.exists():continue
        for value in csv.DictReader(path.open()):
            if value.get(gate) not in ('False','false','0'):continue
            failures.append(dict(name=value.get('name',value.get('trace_name',file)),kind='UNQUALIFIED_INSTRUMENTATION',
                status='REJECTED_LOCAL_TAIL_INTERFERENCE',receipt=file,
                scope='Completed execution can still fail its measurement eligibility gate.'))
    for fragment in (ROOT/'raw').glob('**/*.tmp.*'):
        failures.append(dict(name=fragment.parent.name,kind='INCOMPLETE_WRITE_FRAGMENT',
            status='PRESERVED_NOT_A_VALID_COMPLETION_RECEIPT',receipt=str(fragment.relative_to(ROOT))))
    optional=ROOT/'state/optional_huginn_c32_n32_not_run.json'
    if optional.exists():
        for name in json.loads(optional.read_text())['names']:
            entry=next((v for v in matrix if v['name']==name),None)
            if entry is None:matrix.append(dict(name=name,model='huginn',method='chunked' if 'chunked' in name else 'paced',count=32,worker='rlt_worker',status='NOT_RUN_BUDGET_PRIORITY',complete=False))
            elif not entry['complete']:entry['status']='NOT_RUN_BUDGET_PRIORITY'
    opportunities=list(csv.DictReader((ROOT/'readiness_opportunity.csv').open())) if (ROOT/'readiness_opportunity.csv').exists() else []
    slos=json.loads((ROOT/'configs/frozen_SLO.json').read_text())
    for model in ('ouro','huginn'):
        states={}
        for method in ('full','chunked','paced'):
            name=f'qual32_{model}_full_scoped_v3' if method=='full' else f'qual32_{model}_{method}_c128_v4'
            p=ROOT/'raw'/name/'local_qualification.json'
            state=json.loads(p.read_text()) if p.exists() else None
            states[method]=bool(state['passed'] and state['count']==32) if state else None
        controls=[]
        for name in (f'opportunity_{model}_c128_alignment_v1',f'opportunity_{model}_c248_alignment_v1',f'opportunity_{model}_c248_d_device_v1'):
            rows={v['method']:v for v in opportunities if v.get('name')==name}
            if not all(m in rows for m in ('same_align','cross')):continue
            same,cross=rows['same_align'],rows['cross']
            limit=slos[f'{model}_L{4096 if model=="ouro" else 2048}']
            safe=all(v['output_exact']=='True' and float(v['TPOT_p95_ms'])<=.9*limit['TPOT_ms'] and float(v['ITL_p99_ms'])<=limit['ITL_ms'] for v in (same,cross))
            controls.append(dict(name=name,physical_GPU=cross.get('physical_GPU'),micro_decode_screen=safe,
                cross_over_aligned_same_local=float(cross['local_first_ms'])/float(same['local_first_ms']),
                cross_over_aligned_same_cohort=float(cross['cohort_host_ms'])/float(same['cohort_host_ms']),
                independent_starts=1,warm_repetitions=int(cross['warm_repetitions'])))
        grid=ROOT/'raw'/f'micro_{model}_strict_grid_v5/micro_summary.json'
        grid_value=json.loads(grid.read_text()) if grid.exists() else None
        prefix=grid.parent/'micro_output_parity.json'
        prefix_value=json.loads(prefix.read_text()) if prefix.exists() else None
        grid_exact=bool(grid_value['all_micro_states_exact'] and grid_value['cases']==24
            and prefix_value and prefix_value['passed']) if grid_value else None
        qualified=all(v is True for v in states.values()) and grid_exact is True
        meaningful=any(v['micro_decode_screen'] and (v['cross_over_aligned_same_local']<.95 or v['cross_over_aligned_same_cohort']<.95) for v in controls)
        verdict='INCONCLUSIVE'
        if qualified and controls:
            verdict='MICRO_ONLY' if meaningful else 'NO_GO_LOOP_BATCHING'
        decisions[model]=dict(N32_generic_state_qualification=states,corrected_core_grid_exact=grid_exact,
            aligned_same_depth_controls=controls,verdict=verdict,
            online_cross_depth_benefit='NOT_RUN_PHASE_C_GATE' if not meaningful else 'NOT_MEASURED',
            contiguous_clean_kernel_slack='NOT_MEASURED_QUALIFIED_TRACE_UNAVAILABLE',
            novelty='NOT_ESTABLISHED_BEYOND_NATIVE_MIXED_CDB_AND_GENERIC_ALIGNMENT',
            scope='Bounded PoC expansion decision. It does not prove that all deflection or depth-mixed workloads lack headroom.')
        for method in ('generic_same_depth_online','cross_depth_online'):
            matrix.append(dict(name=f'conditional_{model}_{method}_3x128',model=model,method=method,
                worker='NOT_IMPLEMENTED_CONDITIONAL',count=384,status=decisions[model]['online_cross_depth_benefit'],complete=False))
        matrix.append(dict(name=f'conditional_{model}_persistence512',model=model,method='persistence',count=512,
            worker='CONDITIONAL',status='NOT_RUN_PHASE_C_GATE',complete=False))
    table(ROOT/'configuration_matrix.csv',matrix);table(ROOT/'failure_inventory.csv',failures)
    write(ROOT/'final_model_decisions.json',decisions)
    write(ROOT/'closeout_counts.json',dict(time=time.time(),executed_jobs=len(launched),
        complete_jobs=sum(v['complete'] for v in matrix if 'config' in v),failed_or_blocked_records=len(failures),
        configured_not_run=sum(not v.get('guarded_worker_started',False) for v in matrix if 'config' in v),
        scope='A micro grid is one GPU job with multiple legal-state cells. Conditional online plans are not executed jobs.'))
    print(json.dumps(decisions))


if __name__=='__main__':main()
