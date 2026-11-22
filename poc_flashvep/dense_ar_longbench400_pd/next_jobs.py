"""Freeze the next discriminating tests before their GPU results exist."""
import copy
import hashlib
import json
from .common import ROOT,write,digest
from .experiment import create


def plan(name,jobs,stop=True):
    path=ROOT/'jobs'/(name+'.json')
    value=dict(jobs=jobs,stop_on_failure=stop)
    if path.exists():assert json.loads(path.read_text())==value
    else:write(path,value)


def edge():
    pool=json.loads((ROOT/'data/longbench_balanced400_v1.json').read_text())
    original=next(row for row in pool if row['input_tokens']>8193)
    row=copy.deepcopy(original)
    row.update(prompt_id=original['prompt_id']+'_SHAPE_PREFIX8193',input_tokens=8193,
        input_ids={'qwen8':original['input_ids']['qwen8'][:8193]},
        diagnostic_only=True,context_truncated=True,
        diagnostic_scope='Token-prefix shape stress. Not Main400. No task-quality or serving-gain claim.')
    row['input_ids_SHA256']=hashlib.sha256(json.dumps(row['input_ids']['qwen8'],separators=(',',':')).encode()).hexdigest()
    path=ROOT/'data/diagnostic_query8193.json'
    if path.exists():assert json.loads(path.read_text())==[row]
    else:write(path,[row])
    capsule=ROOT/'references/source_capsules/motivation_baseline_v1.zip'
    common=dict(pair=(2,3),output=16,fixed=True,controlled=True,correctness=True,
        serial=True,wall_timeout=900,serving_timeout=600)
    jobs=[create('Edge8193_Layer4_PreGuard_RED_v1','Layer',path.name,cut=4,
        source_capsule=capsule.name,source_capsule_SHA256=digest(capsule),**common),
        create('Edge8193_Layer4_Guarded_State_v1','Layer',path.name,cut=4,**common),
        create('Edge8193_Full36_Teacher_v1','Native',path.name,pair=(2,3),output=16,
            fixed=True,correctness=True,reference_single=True,serial=True,axis_state_capture=False,
            state_capture=True,numerical_probe=True,numerical_probe_sync=True,kv_layout='HND')]
    plan('query1_graph_edge_v1',jobs,False)


def profile():
    jobs=[]
    for suffix,enabled in [('Clean1',False),('RuntimeNoCopyProfile',True),('Clean2',False)]:
        jobs.append(create('C0_Native_N8_D_'+suffix+'_v4','Native','qualification8.json',
            pair=(6,7),output=16,fixed=True,controlled=True,serial=True,
            kernel_activity=enabled,kernel_activity_roles=['decode'],kernel_activity_seconds=7200,
            operator_correlation=enabled,activity_copies=False,runtime_api_activity=enabled))
    plan('destination_collector_observer_v4',jobs)


def pilots():
    jobs=[]
    for bucket in range(4):
        for output in (8,16,64,128):
            jobs.append(create(f'CapacityC0_I{bucket}_O{output}_Serial16_v1','Native',
                f'input{bucket}_discovery_exact16.json',pair=(4,5),output=output,
                fixed=True,controlled=True,serial=True,qps=20,seed=2026112171))
    plan('mechanism_native_service_pilots_v1',jobs)


def main():
    edge();profile();pilots()
    quality=ROOT/'configs/quality_gate_frozen_v1.json'
    value=dict(reference='CandidateNaturalReference_v2',task_metrics='Pinned official LongBench metrics.',
        paired_task_macro_margin_percentage_points=2.0,confidence=.95,
        request_unit='Context-group bootstrap within task. Server starts remain the performance CI unit.',
        same_geometry='Full valid KV hashes, final hidden and greedy output must match.',
        different_geometry='Report token differences. Require state ownership and coverage. '
            'Require task-score non-inferiority before Primary performance claims.',
        frozen_before_new_partition_Natural400_results=True)
    if quality.exists():assert json.loads(quality.read_text())==value
    else:write(quality,value)
    print('Frozen query1 edge, observer v4, 16 service pilots and quality gate.',flush=True)


if __name__=='__main__':main()
