"""Account for every immutable v2 configuration,including failed and unrun work."""
import collections
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,preserve
from .runner import signature


def cpu_tests():
    result=subprocess.run([str(PYTHON),'-m','unittest','poc_flashvep.dense_ar_output_aware_v2.test_protocol','-v'],
        cwd=REPO,text=True,capture_output=True)
    compile_result=subprocess.run([str(PYTHON),'-m','compileall','-q',str(REPO/'poc_flashvep/dense_ar_output_aware_v2')],
        text=True,capture_output=True)
    write(ROOT/'analysis/cpu_final_regression_v2.json',dict(time=time.time(),passed=result.returncode==0 and compile_result.returncode==0,
        unittest_returncode=result.returncode,unittest_stdout=result.stdout,unittest_stderr=result.stderr,
        compile_returncode=compile_result.returncode,compile_stderr=compile_result.stderr))
    assert result.returncode==compile_result.returncode==0


def inventory():
    preserve();rows=[];statuses=collections.Counter();errors=[]
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg=json.loads(path.read_text())
        if not isinstance(cfg,dict) or not all(k in cfg for k in ('name','dataset','gpus','source_capsule_SHA256')):continue
        name=cfg['name'];folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
        row=dict(name=name,config_SHA256=digest(path),GPU_pair=cfg['gpus'],dataset_SHA256=cfg.get('dataset_SHA256'),
            source_SHA256=cfg['source_capsule_SHA256'],correctness_only=cfg.get('correctness_only'),
            source_credit=cfg.get('continuous_source_credit_limit') if not cfg.get('controlled_pipeline') else cfg.get('source_credit_limit'),
            B=cfg.get('token_budget'),S=cfg.get('max_num_seqs'),QPS=cfg.get('qps'),arrival_seed=cfg.get('seed'))
        if not receipt.exists():
            row['status']='INCOMPLETE_WITHOUT_RECEIPT' if folder.exists() else 'NOT_RUN'
        else:
            result=json.loads(receipt.read_text());row['receipt_SHA256']=digest(receipt)
            row['status']='COMPLETE' if result['complete'] else result.get('status','FAILED_EXECUTION')
            if (folder/'configuration.json').exists():
                if json.loads((folder/'configuration.json').read_text())!=cfg:errors.append(dict(name=name,error='RAW_CONFIG_DIFFERS'))
            if result['complete'] and not cfg.get('correctness_only'):
                proof=json.loads((ROOT/'state'/cfg['qualification_gate_v2']).read_text())
                if not proof['passed'] or proof['signature']!=signature(cfg) or proof['source_capsule_SHA256']!=cfg['source_capsule_SHA256']:
                    errors.append(dict(name=name,error='PERFORMANCE_COMPOSITION_PROOF_MISMATCH'))
            m=ROOT/'analysis'/(name+'_metrics.json');row['metric_SHA256']=digest(m) if m.exists() else None
        rows.append(row);statuses[row['status']]+=1
    gates=[]
    for path in sorted((ROOT/'state').glob('*.json')):
        value=json.loads(path.read_text())
        if isinstance(value,dict) and value.get('evidence') in ('CORRECTNESS_ONLY','FAILED') and 'passed' in value:
            gates.append(dict(file=path.name,SHA256=digest(path),passed=value['passed'],error=value.get('error')))
    plan_status=[]
    byname={r['name']:r['status'] for r in rows}
    for path in sorted((ROOT/'jobs').glob('*.json')):
        data=json.loads(path.read_text());names=set(n for lane in data.get('lanes',[]) for n in lane)
        if not names:continue
        plan_status.append(dict(plan=path.name,SHA256=digest(path),unique_jobs=len(names),
            status_counts=dict(collections.Counter(byname.get(n,'NO_CONFIG') for n in names))))
    result=dict(time=time.time(),passed=not errors,status_counts=dict(statuses),jobs=rows,composition_gates=gates,plans=plan_status,errors=errors,
        scope='Inventory every prospective configuration. COMPLETE is execution completion,not correctness or scientific success. '
            'Strict-state failures can occur after a completed execution. Keep those failed composition gates separate. '
            'NOT_RUN includes frozen alternatives and budget/safety stops. Never infer a measured value for an unrun point. '
            'Raw directories and receipts remain unchanged. This inventory does not promote a failed gate or an incomplete campaign.')
    write(ROOT/'analysis/final_configuration_and_gate_inventory_v2.json',result)
    from .c0_analysis import csv_save
    csv_save(ROOT/'figures/final_configuration_execution_inventory_v2.csv',rows)
    csv_save(ROOT/'figures/final_plan_execution_inventory_v2.csv',plan_status)
    preserve();print('V2_CONFIGURATION_ACCOUNTING',dict(statuses),'ERRORS',len(errors),flush=True);assert not errors
    return result


if __name__=='__main__':cpu_tests();inventory()
