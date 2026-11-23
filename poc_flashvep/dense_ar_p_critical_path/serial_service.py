"""Measure serving operators with one request alive and no model-call observer."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind
from .bootstrap import baseline

PLAN='serial_serving_operator_cpc_v1.json'


def prepare():
    runner=bind();data='qualification32.json';count=32;tests=[];jobs=[]
    for index in range(3):
        order=(('Layer','StockToken','Hybrid'),('Hybrid','StockToken','Layer'),('StockToken','Layer','Hybrid'))[index]
        for label in order:
            names=[]
            for stage in ('Clean1','Profile','Clean2'):
                name=f'PC_SerialService_{label}_s{index}_{stage}'
                cfg=baseline(label,name,[0,1],save_cfg=False)
                cfg.update(dataset=data,dataset_SHA256=digest(ROOT/'data'/data),count=count,
                    arrival_offsets=[0.]*count,serial=True,ignore_eos=True,output_cap=16,
                    replay_output_lengths=[16]*count,diagnostic_controlled_output_replay=True,
                    kernel_activity=stage=='Profile',kernel_activity_roles=['prefill','decode'],
                    collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,activity_copies=False,
                    operator_correlation=False,graph_enclosures=False,activity_delay_seconds=0,activity_ROI_seconds=None,
                    kernel_activity_seconds=86400,
                    observer_scope='Original serving API and kernels. Serial request lifetime. No model-call event observer. '
                        'This is a Batch1 mechanism control,not an online Poisson capacity experiment. '
                        'Both P and D kernel timelines are captured. Copies,SM and HBM are not measured.')
                if cfg.get('axis_kind')=='Token':
                    from ..dense_ar_output_aware_v2.work_contract import regenerate
                    regenerate(cfg,json.loads((ROOT/'data'/data).read_text()))
                runner.save(cfg);names.append(name);jobs.append(name)
            tests.append(dict(label=label,start=index,jobs=names,pair=[0,1],count=count,
                family='SERIAL_C3_BATCH1_SERVICE_CONTROL'))
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),lanes=[jobs,[],[],[]],tests=tests,stop_on_failure=True,
        selection='Use the frozen qualification32 requests. Do not select them by measured serving performance.',
        scope='Nine predeclared CPC triplets. Original full-source model and partition paths stay unchanged. '
            'One request completes before the next request arrives. Actual P,D suffix and D decode batches must be1. '
            'Serial arrival times are execution-dependent. Do not claim the same Poisson trace across these controls. '
            'Layer2,StockToken8 and Hybrid2/f1/8 remain best-vs-best,not iso-work. '
            'Primary C3 N3 comparisons remain separate. No host delay is injected.'))
    log=ROOT/'logs/serial_serving_operator_cpc_v1_controller.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.serial_service','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/serial_serving_operator_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def prepare_matched():
    runner=bind();tests=[];jobs=[]
    for label in ('Layer8','StockToken','HybridIso8Approx'):
        names=[]
        for stage in ('Clean1','Profile','Clean2'):
            name=f'PC_SerialMatched_{label}_s0_{stage}'
            if label=='StockToken':cfg=baseline(label,name,[0,1],save_cfg=False)
            else:
                cfg=json.loads((ROOT/'configs'/f'PC_Balance_{label}_Natural400.json').read_text())
                cfg['qualification_gate_v2']='pc_balance_'+label.lower()+'_composition.json'
            dataset='qualification32.json';count=32
            cfg.update(name=name,gpus=[0,1],dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=count,
                serial=True,arrival_offsets=[0.]*count,ignore_eos=True,output_cap=16,
                replay_output_lengths=[16]*count,diagnostic_controlled_output_replay=True,
                correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
                kernel_activity=stage=='Profile',kernel_activity_roles=['prefill','decode'],
                collector_namespace='kernel_only_collector_v3',runtime_api_activity=False,activity_copies=False,
                operator_correlation=False,graph_enclosures=False,activity_delay_seconds=0,activity_ROI_seconds=None,
                kernel_activity_seconds=86400,
                observer_scope='Serial32 actual-Batch1 serving control. One CPC triplet per geometry. '
                    'Compare Layer8,StockToken WorkCut8 and Hybrid2/f3/17 at nominal P28L/D8L rows. '
                    'Audit actual16-token alignment and executed query/layer coverage. '
                    'No model-call observer. This is not a Poisson online superiority experiment.')
            if cfg.get('axis_kind')=='Token':
                from ..dense_ar_output_aware_v2.work_contract import regenerate
                regenerate(cfg,json.loads((ROOT/'data'/dataset).read_text()))
            runner.save(cfg);names.append(name);jobs.append(name)
        tests.append(dict(label=label,start=0,jobs=names,pair=[0,1],count=32,family='SERIAL_NOMINAL_ISO_WORK_SERVICE_CONTROL'))
    plan='serial_matched_work_operator_cpc_v1.json';path=ROOT/'jobs'/plan;assert not path.exists()
    write(path,dict(time=time.time(),lanes=[jobs,[],[],[]],tests=tests,stop_on_failure=True,
        scope='Three predefined triplets,not a full cut sweep. Use the frozen32 prompts and fixed Output16. '
            'Actual matching is approximate under16-token alignment. Preserve the signed row errors. '
            'A qualified single triplet does not establish a three-start operator effect. '
            'Both P/D kernels use the original serving APIs. No unverified decode-safe admission is added.'))
    from .bootstrap import launch
    return launch(plan)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':print(prepare(),flush=True)
    elif sys.argv[1]=='matched':print(prepare_matched(),flush=True)
    else:bind().lane(PLAN,0)
