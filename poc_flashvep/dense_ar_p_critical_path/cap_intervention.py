"""Test a D-cap-only response to the measured geometry bottleneck shift."""
import json
import subprocess
import time
import numpy as np
from .common import ROOT,PRIOR,REPO,PYTHON,write,digest,bind,dispatch_allowed
from .bootstrap import baseline,launch
from .native_capture_fix import capture_flags

QUAL='p_geometry_D_cap4_qualification_v1.json'
DISCOVERY='p_geometry_D_cap4_discovery_v1.json'
CONFIRM='p_geometry_D_cap4_confirmation_v1.json'
LABELS=('NativeCommon4','StockCommon4','Layer8Cap4','HybridIsoCap4')


def gate(label):return 'pc_geometry_D_cap4_'+label.lower()+'.json'


def prepare():
    runner=bind();items=[]
    parents={'NativeCommon4':'PC_C3_EqualReplay_s0_NativeCommon2',
        'StockCommon4':'PC_C3_EqualReplay_s0_StockToken',
        'Layer8Cap4':'PC_Balance_Layer8_Natural400','HybridIsoCap4':'PC_Balance_HybridIso8Approx_Natural400'}
    for label,parent in parents.items():
        original=json.loads((ROOT/'configs'/(parent+'.json')).read_text());jobs=[]
        for count,data in ((32,'qualification32.json'),(400,'longbench_balanced400_v2.json')):
            value=dict(original);name=f'PC_DCap4_{label}_Natural{count}'
            value.update(name=name,gpus=[0,1],dataset=data,dataset_SHA256=digest(ROOT/'data'/data),count=count,
                correctness_only=True,axis_state_capture=label!='NativeCommon4',state_capture=label=='NativeCommon4',
                axis_capture_hidden_last_only=True,axis_import_hash_after_model=True,gpu_state_digest=True,
                verify_GPU_digest_with_CPU=False,D_running_admission_cap=4,ignore_eos=False,output_cap=2048,
                P_event_timing=False,kernel_activity=False,coordinator_trace=False,
                qps=3.2,seed=2026112400,
                arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(2026112400).exponential(1/3.2,count-1))].tolist())
            for key in ('replay_output_lengths','diagnostic_controlled_output_replay','qualification_gate_v2'):value.pop(key,None)
            if value.get('axis_kind')=='Token':
                from ..dense_ar_output_aware_v2.work_contract import regenerate
                regenerate(value,json.loads((ROOT/'data'/data).read_text()))
            runner.save(value);jobs.append(name)
        arm='NativeCommon' if label=='NativeCommon4' else 'StockToken' if label=='StockCommon4' else 'Layer' if label=='Layer8Cap4' else 'Hybrid'
        items.append(dict(label=label,arm=arm,lane=0,pair=[0,1],teacher=None,strict=None,jobs=jobs,
            gate=gate(label),signature=runner.signature(value),source_capsule_SHA256=value['source_capsule_SHA256']))
    path=ROOT/'configs'/QUAL;assert not path.exists()
    write(path,dict(time=time.time(),items=items,state_prefix='pc_geometry_D_cap4',source_capsule_SHA256=None,
        observation='Equal-output N3 Layer2→Layer8 reduced P arrival→ready ratio to0.60567 but raised TTFT p95 ratio to1.07721. '
            'D Prepare→First Worker increased194.930ms. This motivates a separate admission response,not a new cut sweep.',
        hypothesis='A larger D cohort may amortize continuation and decode scheduling costs after moving more work to D.',
        competing_hypothesis='D is compute-limited. A larger cap cannot remove the bottleneck and may worsen decode tails.',
        intervention='Change only D running admission cap2→4 within Layer8/Hybrid iso geometry. Keep same-axis source16/8 and compiler. '
            'Retain strong Stock cap0/source0. Also measure Stock cap4/source0 and exact Common-D-policy Native cap4/source0.',
        screening='Discovery208 only. Select a geometry candidate if TTFT p95≤its cap2 control and absolute decode attainment≥.95. '
            'Also require point ITL p99 and TPOT p95≤1.05 of NativeStrong and Common4. N1 screening is not safety confirmation.',
        correctness_scope='D cap changes actual scheduling geometry,not model operators. Reuse original numerical-family teacher evidence. '
            'Fresh32/400 valid-state,full36,import,lifetime and unchanged2pp task-quality gates are mandatory. '
            'No removal of the existing whole-cohort safety policy and no active-decode insertion.'))
    log=ROOT/'logs/p_geometry_D_cap4_controller_v1.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.cap_intervention','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/p_geometry_D_cap4_controller_v1.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def template(label):
    if label in LABELS:
        cfg=json.loads((ROOT/'configs'/f'PC_DCap4_{label}_Natural400.json').read_text());cfg['qualification_gate_v2']=gate(label);return cfg
    if label=='OriginalLayer':
        cfg=json.loads((ROOT/'configs/PC_Balance_Layer8_Natural400.json').read_text());cfg['qualification_gate_v2']='pc_balance_layer8_composition.json';return cfg
    if label=='OriginalHybrid':
        cfg=json.loads((ROOT/'configs/PC_Balance_HybridIso8Approx_Natural400.json').read_text());cfg['qualification_gate_v2']='pc_balance_hybridiso8approx_composition.json';return cfg
    return baseline(label,'PC_DCap4_Template_'+label,[0,1],save_cfg=False)


def timing(label,name,data_name,pair,seed,natural=False):
    runner=bind();data=json.loads((ROOT/'data'/data_name).read_text());count=len(data);cfg=template(label)
    cfg.update(name=name,gpus=pair,dataset=data_name,dataset_SHA256=digest(ROOT/'data'/data_name),count=count,
        correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
        P_event_timing=False,kernel_activity=False,coordinator_trace=False,
        ignore_eos=not natural,output_cap=2048,qps=2.,seed=seed,
        arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(.5,count-1))].tolist())
    if natural:
        cfg.pop('replay_output_lengths',None);cfg.pop('diagnostic_controlled_output_replay',None)
    else:
        reference=json.loads((PRIOR/'state/official_compiled_c1_reference_complete_v2.json').read_text())
        refs={row['prompt_id']:row['output_tokens'] for row in map(json.loads,(PRIOR/'raw'/reference['name']/'requests.jsonl').read_text().splitlines())}
        cfg.update(diagnostic_controlled_output_replay=True,replay_output_lengths=[refs[row['prompt_id']] for row in data])
    if cfg.get('axis_kind')=='Token':
        from ..dense_ar_output_aware_v2.work_contract import regenerate
        regenerate(cfg,data)
    runner.save(cfg);return cfg


def run():
    runner=bind()
    from ..dense_ar_output_aware_v2.c3_family import run as qualify
    qualify(0,QUAL)
    proofs={label:json.loads((ROOT/'state'/gate(label)).read_text()) for label in LABELS}
    if not proofs['NativeCommon4']['passed']:
        write(ROOT/'state/p_geometry_D_cap4_not_run.json',dict(time=time.time(),status='NOT_RUN',reason='Common4 Native failed correctness',proofs=proofs));return
    labels=['NativeStrong','NativeCommon2','NativeCommon4','StockToken','OriginalLayer','OriginalHybrid']+[label for label in LABELS[1:] if proofs[label]['passed']]
    arms={label:'PC_DCap4_Discovery208_'+label for label in labels};jobs=[]
    for label,name in arms.items():timing(label,name,'balanced400_v2_discovery.json',[0,1],2026112401);jobs.append(name)
    lanes=[jobs,[],[],[]];write(ROOT/'jobs'/DISCOVERY,dict(time=time.time(),lanes=lanes,arms=arms,stop_on_failure=False))
    runner.lane(DISCOVERY,0)
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    measured={label:summarize(name) for label,name in arms.items()
        if (ROOT/'raw'/name/'dispatcher_result.json').exists() and json.loads((ROOT/'raw'/name/'dispatcher_result.json').read_text())['complete']}
    selected=[];screens=[]
    for candidate,control in (('Layer8Cap4','OriginalLayer'),('HybridIsoCap4','OriginalHybrid')):
        if not all(label in measured for label in (candidate,control,'NativeStrong','NativeCommon4')):continue
        value=measured[candidate];ratio=value['TTFT_p95_ms']/measured[control]['TTFT_p95_ms']
        decode=all(value[key]<=1.05*measured[ref][key] for key in ('ITL_p99_ms','TPOT_p95_ms') for ref in ('NativeStrong','NativeCommon4'))
        passed=ratio<=1 and value['decode_SLO_attainment']>=.95 and decode
        screens.append(dict(candidate=candidate,control=control,TTFT_ratio=ratio,point_decode_pass=decode,
            absolute_decode_attainment=value['decode_SLO_attainment'],selected=passed))
        if passed:selected.append(candidate)
    selection=ROOT/'state/p_geometry_D_cap4_selection_frozen.json';assert not selection.exists()
    write(selection,dict(time=time.time(),selected=selected,screens=screens,all_measured=measured,
        discovery_plan_SHA256=digest(ROOT/'jobs'/DISCOVERY),scope='Freeze before any new heldout result. No failed cell is removed.'))
    if not selected:
        write(ROOT/'state/p_geometry_D_cap4_confirmation_not_run.json',dict(time=time.time(),status='NOT_RUN',reason='Neither predefined cap4 geometry passed the Discovery screen.'));return
    mapping={label:label for label in ('NativeStrong','NativeCommon2','NativeCommon4','StockToken')}
    if 'StockCommon4' in measured:mapping['StockCommon4']='StockCommon4'
    for candidate in selected:
        axis='Layer' if candidate=='Layer8Cap4' else 'Hybrid';mapping['Original'+axis]='Original'+axis;mapping['Optimized'+axis]=candidate
    blocks=[];lanes=[[],[],[],[]]
    for mode,pair,lane in (('EqualReplay',[0,1],0),('Natural',[4,5],2)):
        for index,seed in enumerate((2026112421,2026112422,2026112423)):
            names={}
            for label,parent in mapping.items():
                name=f'PC_DCap4_{mode}_s{index}_{label}';timing(parent,name,'balanced400_v2_heldout.json',pair,seed,mode=='Natural');names[label]=name
            order=list(mapping)
            if index==1:order.reverse()
            elif index==2:order=order[2:]+order[:2]
            lanes[lane].extend(names[label] for label in order)
            blocks.append(dict(mode=mode,start=index,seed=seed,qps=2.,pair=pair,arms=names,order=order))
    write(ROOT/'jobs'/CONFIRM,dict(time=time.time(),blocks=blocks,arms=list(mapping),lanes=lanes,stop_on_failure=False,
        additional_decode_references=['NativeCommon4'],extra_pairs=[['StockCommon4','StockToken']],
        selection_SHA256=digest(selection),analysis_scope='D-cap-only intervention at the already frozen Layer8/Hybrid iso geometries. '
            'Optimized means D cap4,Original means D cap2. Same-axis source credit,work and compiler remain unchanged. '
            'Strong Stock cap0/source0 stays intact. NativeStrong,NativeCommon2 and exact NativeCommon4 are separate controls. '
            'This family is not pooled with credit,geometry or coordinator studies. Natural output work may differ. '
            'No active-decode insertion or future safe-window prediction is added.'))
    print('D_CAP4_CONFIRMATION_LAUNCHED',launch(CONFIRM),flush=True)


def analyze_discovery():
    bind()
    from .critical_analysis import csv_save
    path=ROOT/'state/p_geometry_D_cap4_selection_frozen.json'
    selection=json.loads(path.read_text());rows=[]
    plan=json.loads((ROOT/'jobs'/DISCOVERY).read_text())
    for label,metric in selection['all_measured'].items():
        name=plan['arms'][label];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        row=dict(metric)
        row.update(label=label,name=name,independent_starts=1,QPS=cfg['qps'],seed=cfg['seed'],
            source_credit=cfg.get('continuous_source_credit_limit',0),D_cap=cfg.get('D_running_admission_cap'),
            axis_cut=cfg.get('axis_cut'),hybrid_suffix_fraction=cfg.get('hybrid_suffix_fraction'),
            config_SHA256=digest(ROOT/'configs'/(name+'.json')),dataset_SHA256=cfg['dataset_SHA256'],
            source_SHA256=cfg['source_capsule_SHA256'],GPU_pair=cfg['gpus'],
            evidence='MEASURED_ONLINE_DISCOVERY_SINGLE_START')
        rows.append(row)
    value=dict(time=time.time(),rows=rows,screens=selection['screens'],selected=selection['selected'],
        selection_SHA256=digest(path),plan_SHA256=digest(ROOT/'jobs'/DISCOVERY),
        confirmation='NOT_RUN_BY_PREDECLARED_SCREEN' if not selection['selected'] else 'SEE_SEPARATE_CONFIRMATION',
        scope='Keep all nine measured Discovery arms. Layer8 and Hybrid-iso cap2→4 are one-factor interventions. '
            'Strong Stock cap0/source0 is not replaced by cap4 Stock. All208 requests emit the same independent C1 replay length. '
            'N1 point safety is screening only,not paired decode noninferiority or a confirmed serving improvement.')
    write(ROOT/'analysis/p_geometry_D_cap4_discovery_analysis.json',value)
    csv_save(ROOT/'figures/p_geometry_D_cap4_discovery.csv',rows)
    csv_save(ROOT/'figures/p_geometry_D_cap4_discovery_screens.csv',selection['screens'])
    print('D_CAP4_DISCOVERY_ANALYZED',len(rows),'arms,selected',selection['selected'],flush=True)
    return value


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':print(prepare(),flush=True)
    elif sys.argv[1]=='discovery-analysis':analyze_discovery()
    elif sys.argv[1]=='analyze':
        from .confirmation_analysis import analyze
        analyze(CONFIRM,'p_geometry_D_cap4_confirmation')
    else:run()
