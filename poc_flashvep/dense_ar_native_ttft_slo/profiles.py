"""Predeclare serial Native-relative P/D kernel controls with CPC gates."""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from .common import ROOT,PRIOR,REPO,PYTHON,write,digest,preserve,resources
from .runtime import bind,copy_data,dispatch_allowed

PLAN='serial_serving_operator_cpc_v2.json'


def collector():
    source=PRIOR/'references/kernel_only_collector_v3'
    record=json.loads((source/'build.json').read_text())
    target=ROOT/'references/kernel_only_collector_v3';target.mkdir(parents=True,exist_ok=True)
    for key in ('source','library'):
        original=Path(record[key]);destination=target/original.name
        if not destination.exists():shutil.copyfile(original,destination)
        assert digest(destination)==record[key+'_SHA256']
        record[key]=str(destination)
    write(target/'build.json',record)


def prepare():
    runner=bind();collector();data='qualification32.json';copy_data(data);tests=[];jobs=[]
    qualification=json.loads((ROOT/'configs/root_cause_policy_qualification_v1.json').read_text())
    templates={row['label']:row for row in qualification['items']}
    labels=('NativeStrong','StockToken','Layer','Hybrid')
    for index in range(3):
        order=labels[index:]+labels[:index]
        for label in order:
            parent=templates[label];cfg=json.loads((ROOT/'configs'/(parent['steps']['Natural400']+'.json')).read_text())
            names=[]
            for stage in ('Clean1','Profile','Clean2'):
                name=f'NS_Serial_{label}_s{index}_{stage}'
                value=dict(cfg,name=name,gpus=[0,1],dataset=data,dataset_SHA256=digest(ROOT/'data'/data),count=32,
                    serial=True,arrival_offsets=[0.]*32,ignore_eos=True,output_cap=16,
                    replay_output_lengths=[16]*32,diagnostic_controlled_output_replay=True,
                    correctness_only=False,state_capture=False,axis_state_capture=False,gpu_state_digest=False,
                    qualification_gate_v2=parent['gate'],kernel_activity=stage=='Profile',
                    kernel_activity_roles=['prefill','decode'],collector_namespace='kernel_only_collector_v3',
                    runtime_api_activity=False,activity_copies=False,operator_correlation=False,graph_enclosures=False,
                    activity_delay_seconds=0,activity_ROI_seconds=None,kernel_activity_seconds=86400,
                    observer_scope='Serial32 actual serving APIs. One request alive. No model-call observer. '
                        'Both P/D CUDA kernel activities;transfer DMA,SM and HBM are not measured.')
                from ..dense_ar_output_aware_v2.work_contract import regenerate
                regenerate(value,json.loads((ROOT/'data'/data).read_text()))
                assert runner.signature(value)==parent['signature'],label
                runner.save(value);names.append(name);jobs.append(name)
            tests.append(dict(label=label,start=index,jobs=names,pair=[0,1],count=32,
                family='SERIAL_NATIVE_RELATIVE_SERVICE_CONTROL'))
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),lanes=[jobs,[],[],[]],tests=tests,
        scope='Frozen qualification32 prompts and exactly16 output tokens. This is a serial actual-Batch1 mechanism control. '
            'It is not the Main400 Natural online result or sustainable capacity. Three prospective triplets per arm. '
            'A failed5% CPC gate stops further profiling of that arm;retain remaining cells as NOT_RUN. '
            'Do not repeat a failed triplet until it passes. Kernel union and operator partition do not equal TTFT. '
            'Native and split P/D work differ as prescribed;measure saved and added service,not matched FLOPs.'))
    log=ROOT/'logs/serial_native_relative_profile_controller_v2.log'
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_native_ttft_slo.profiles','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/serial_native_relative_profile_controller_v2.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def cpc(test):
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from .reproduction import records
    metrics=[summarize(name) for name in test['jobs']]
    raw=[records(ROOT/'raw'/name/'requests.jsonl') for name in test['jobs']]
    gold=[(r['prompt_id'],r['output_tokens']) for r in sorted(raw[0],key=lambda r:r['sample_id'])]
    exact=all([(r['prompt_id'],r['output_tokens']) for r in sorted(rows,key=lambda r:r['sample_id'])]==gold for rows in raw)
    fields=('TTFT_p95_ms','TTFT_mean_ms','ITL_p99_ms','TPOT_p95_ms','mean_TBT_ms')
    ratios={field:metrics[1][field]/((metrics[0][field]+metrics[2][field])/2) for field in fields}
    counters={role:json.loads((ROOT/'raw'/test['jobs'][1]/('activity_'+role)/'activity_counters.json').read_text())
        for role in ('prefill','decode')}
    batch1=all(role['maximum_requests']==1 for metric in metrics for role in metric['actual_geometry'].values())
    passed=exact and batch1 and all(abs(value-1)<=.05 for value in ratios.values()) and all(value['passed'] for value in counters.values())
    return dict(passed=passed,observer_ratios=ratios,exact_work=exact,actual_batch1=batch1,counters=counters)


def run():
    bind()
    from ..dense_ar_p_critical_path.stable_source import install
    runner=install();plan=json.loads((ROOT/'jobs'/PLAN).read_text());owner=resources();results=[];blocked=set()
    gates=['ns_'+label+'_composition_v1.json' for label in ('nativestrong','stocktoken','layer','hybrid')]
    # Do not mix correctness/compilation contention with the isolated service control.
    while True:
        ready=all((ROOT/'state'/f'policy_qualification_lane{lane}_complete_v1.json').exists() for lane in range(4))
        ready=ready and (ROOT/'state/native_common0_reference_addendum_complete_v1.json').exists()
        if ready or not dispatch_allowed():break
        time.sleep(10)
    all_baselines=all((ROOT/'state'/name).exists() and json.loads((ROOT/'state'/name).read_text())['passed'] for name in gates)
    with (ROOT/'state/pair_0_1.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for test in plan['tests']:
                row=dict(label=test['label'],start=test['start'],jobs=test['jobs'])
                if not all_baselines or test['label'] in blocked or not dispatch_allowed():
                    row.update(status='NOT_RUN',reason='BASELINE_GATE_OR_PREVIOUS_OBSERVER_FAILURE_OR_BUDGET');results.append(row);continue
                try:
                    for name in test['jobs']:
                        assert not (ROOT/'raw'/name).exists(),'Do not restart a live or failed profile'
                        owner.stop([0,1]);assert runner.execute(name),name
                    gate=cpc(test);row.update(status='COMPLETE',CPC=gate)
                    if not gate['passed']:blocked.add(test['label']);row['evidence']='DIAGNOSTIC'
                except Exception as error:
                    blocked.add(test['label']);row.update(status='FAILED',error=repr(error),traceback=traceback.format_exc())
                results.append(row)
                write(ROOT/'state/serial_native_relative_profile_progress_v2.json',dict(time=time.time(),results=results))
                print('NS_SERIAL_CPC',json.dumps(row),flush=True)
        finally:owner.restore([0,1]);preserve()
    write(ROOT/'state/serial_native_relative_profile_complete_v2.json',dict(time=time.time(),results=results,complete=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    print(prepare() if args.action=='prepare' else run(),flush=True)
