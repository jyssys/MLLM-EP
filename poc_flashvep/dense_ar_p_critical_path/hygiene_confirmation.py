"""Use qualified common coordinator code without weakening strong Stock."""
import json
import subprocess
import time
import numpy as np
from .common import ROOT,REPO,PYTHON,write,digest,bind,dispatch_allowed

PLAN='coordinator_hygiene_c3_confirmation_r2.json'
STABLE_STOCK=False
FIXED_PORTS=False
MAPPING={'NativeStrong':'NativeStrong','NativeCommon2':'NativeCommon2','StockToken':'StockToken',
    'OriginalLayer':'Layer','OriginalHybrid':'Hybrid','OptimizedLayer':'Layer8','OptimizedHybrid':'HybridIso8Approx',
    'HybridGCControl':'HybridGCControl','NativeCommon16':'NativeCommon16','StockCommon16':'StockCommon16',
    'HybridCommon16':'HybridCommon16','HybridIsoCommon16':'HybridIsoCommon16'}


def gate_name(label):
    if FIXED_PORTS:
        from .port_qualification import gate
        return gate(label)
    if STABLE_STOCK and label=='StockToken':return 'pc_coord_stocktoken_stable_source_composition_v5.json'
    if label in ('NativeStrong','NativeCommon2','NativeCommon16'):suffix='composition_state_capture_v3.json'
    elif label in ('StockToken','Layer','Hybrid','Layer8','HybridIso8Approx'):suffix='composition_same_source_v4.json'
    else:suffix='composition_r2.json'
    return 'pc_coord_'+label.lower()+'_'+suffix


def parent_name(label):
    if FIXED_PORTS:return 'PC_PortQual_'+label+'_Natural400'
    if STABLE_STOCK and label=='StockToken':return 'PC_StableSource_StockToken_Natural400'
    stem='PC_CommonPolicy_' if label in ('NativeCommon16','StockCommon16','HybridCommon16','HybridIsoCommon16') else 'PC_CoordQual_'
    return stem+label+'_Natural400'


def prepare_after_gates():
    runner=bind();proofs={}
    for label in MAPPING.values():
        path=ROOT/'state'/gate_name(label)
        while not path.exists():
            if not dispatch_allowed():return None
            time.sleep(15)
        proofs[label]=json.loads(path.read_text())
    invalid={label:proof for label,proof in proofs.items() if not proof['passed']}
    if any(label in invalid for label in ('NativeStrong','NativeCommon2','StockToken')):
        marker=('coordinator_hygiene_fixed_ports_not_run_v6.json' if FIXED_PORTS else
            'coordinator_hygiene_stable_stock_not_run_v5.json' if STABLE_STOCK else 'coordinator_hygiene_confirmation_not_run.json')
        write(ROOT/'state'/marker,dict(time=time.time(),status='NOT_RUN',
            reason='A required strong baseline failed its coordinator composition gate. Do not substitute a weaker baseline.',invalid=invalid,
            qualification_gate_SHA256={label:digest(ROOT/'state'/gate_name(label)) for label in proofs}))
        return None
    data_name='balanced400_v2_heldout.json';data=json.loads((ROOT/'data'/data_name).read_text());count=len(data)
    lengths=json.loads((ROOT/'configs/PC_C3_EqualReplay_s0_OriginalLayer.json').read_text())['replay_output_lengths']
    blocks=[];lanes=[[],[],[],[]]
    for mode,pair,lane in (('EqualReplay',[2,3],1),('Natural',[6,7],3)):
        labels=list(MAPPING)
        if mode=='Natural':labels=[label for label in labels if label not in ('HybridGCControl','HybridCommon16')]
        for index,seed in enumerate((2026112391,2026112392,2026112393)):
            names={}
            for label in labels:
                parent=MAPPING[label];cfg=json.loads((ROOT/'configs'/(parent_name(parent)+'.json')).read_text())
                prefix='PC_CoordPortV6' if FIXED_PORTS else 'PC_CoordHygiene'
                name=f'{prefix}_{mode}_s{index}_{label}'
                cfg.update(name=name,gpus=pair,dataset=data_name,dataset_SHA256=digest(ROOT/'data'/data_name),count=count,
                    correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
                    qps=2.,seed=seed,arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(.5,count-1))].tolist(),
                    ignore_eos=mode=='EqualReplay',output_cap=2048,confirmation=True,
                    qualification_gate_v2=gate_name(parent),
                    request_holdout_scope='The coordinator stall was diagnosed on an earlier heldout arrival seed. '
                        'This confirmation uses new arrival seeds,not unseen requests. Static geometry selection used only Discovery208.')
                if mode=='EqualReplay':cfg.update(replay_output_lengths=lengths,diagnostic_controlled_output_replay=True)
                else:cfg.pop('replay_output_lengths',None);cfg.pop('diagnostic_controlled_output_replay',None)
                if cfg.get('axis_kind')=='Token':
                    from ..dense_ar_output_aware_v2.work_contract import regenerate
                    regenerate(cfg,data)
                if parent not in invalid:assert runner.signature(cfg)==proofs[parent]['signature'],label
                runner.save(cfg);names[label]=name
            order=list(labels)
            if index==1:order.reverse()
            elif index==2:order=order[3:]+order[:3]
            lanes[lane].extend(names[label] for label in order if MAPPING[label] not in invalid)
            blocks.append(dict(mode=mode,start=index,seed=seed,qps=2.,pair=pair,arms=names,order=order))
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),blocks=blocks,arms=list(MAPPING),lanes=lanes,stop_on_failure=True,
        invalid_compositions=invalid,
        additional_decode_references=['NativeCommon16'],
        extra_pairs=[['OriginalHybrid','HybridGCControl'],['StockCommon16','StockToken'],
            ['HybridCommon16','OriginalHybrid'],['HybridIsoCommon16','OptimizedHybrid'],['NativeCommon16','NativeCommon2'],
            ['OriginalLayer','StockCommon16'],['OptimizedLayer','StockCommon16'],
            ['HybridCommon16','StockCommon16'],['HybridIsoCommon16','StockCommon16'],
            ['OptimizedLayer','NativeCommon16'],['HybridIsoCommon16','NativeCommon16']],
        analysis_scope='Common coordinator hygiene. Evidence writes and coordinator cyclic GC occur outside measured token delivery. '
            'Client timestamps remain real. Workers retain their qualified original model,attention and transfer paths. '
            'Strong Stock stays compiled WorkCut8,source0,D cap0. NativeStrong and Common-D-policy Native remain. '
            'Common16 controls use logical source16,D cap2 and the same registration/remote-read deferral. '
            'Same policy does not imply the same actual GPU batch or hidden-buffer ownership. '
            'Optimized labels denote the frozen Last8 and Hybrid2/f3/17 geometry controls. '
            'Only OriginalHybrid/HybridGCControl changes coordinator GC timing. Both already defer evidence writes. '
            'The cohort is finite. No sustainable capacity or safe-window predictor is claimed. '
            'Natural has ten arms. The GC-only and original-Hybrid credit-only controls are limited to equal-output N3. '
            'Invalid compositions remain in the registry. No performance is measured for those arms. '
            +('All arms use the same frozen NIXL control-address policy and verified stable source paths. '
                'Actual cache-root equality is checked separately from model-state equality. '
                'This is a new runtime family. Do not pool it with earlier families. ' if FIXED_PORTS else '')
            +('Stock alone also uses a byte-verified stable source path to retain official compiler Artifact reuse. '
                'This setup repair is separately qualified. Keep the original compiled Stock family and its failed new-GC strict result. '
                'This is not an axis-only or common-Artifact-policy comparison.' if STABLE_STOCK else '')))
    return path


def run(version=1):
    global PLAN,STABLE_STOCK,FIXED_PORTS
    if version==5:
        STABLE_STOCK=True;PLAN='coordinator_hygiene_c3_confirmation_stable_stock_v5.json'
    elif version==6:
        FIXED_PORTS=True;PLAN='coordinator_hygiene_c3_confirmation_fixed_ports_v6.json'
    bind();path=prepare_after_gates()
    if path is None:return
    from .bootstrap import launch
    print(launch(path.name),flush=True)


def launch_controller(version=1):
    bind();suffix='' if version==1 else '_v'+str(version)
    log=ROOT/'logs'/('coordinator_hygiene_confirmation_controller'+suffix+'.log');assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.hygiene_confirmation','run',str(version)],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state'/('coordinator_hygiene_confirmation_controller'+suffix+'.json'),dict(time=time.time(),pid=child.pid,version=version))
    return child.pid


if __name__=='__main__':
    import sys
    if sys.argv[1]=='launch':print(launch_controller(int(sys.argv[2]) if len(sys.argv)>2 else 1),flush=True)
    elif sys.argv[1]=='analyze':
        from .confirmation_analysis import analyze
        analyze(PLAN,'coordinator_hygiene_confirmation')
    else:run(int(sys.argv[2]) if len(sys.argv)>2 else 1)
