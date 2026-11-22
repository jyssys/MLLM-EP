"""Validate the same host fix at the other frozen primary load."""
import argparse
import json
import time
import numpy as np
from .common import ROOT, write, digest
from .final_confirmation_jobs import clone
from .final_continuous_selection import policy
from .registration_deferral_campaign import registration_signature, run_plan
from .run import save


PLAN='request_registration_q28_extension_frozen_v1.json'


def build():
    path=ROOT/'configs'/PLAN
    assert not path.exists(), 'Do not retune an extension after its measurements'
    first=json.loads((ROOT/'configs/request_registration_deferral_frozen_v1.json').read_text())
    main=json.loads((ROOT/'configs/final_continuous_execution_plan_v1.json').read_text())
    cell=next(c for c in main['cells'] if c['workload']=='Main400' and c['QPS']==2.8 and c['start']==0)
    original={a:json.loads((ROOT/'configs'/(n+'.json')).read_text()) for a,n in cell['jobs'].items()}
    def native(wanted):
        options=[c for a,c in original.items() if a.startswith('Native') and policy(c)==wanted]
        assert len(options)==1
        return options[0]
    native_read=native((True,True,False,None,False))
    native_cap2=native((True,True,False,2,False))
    prototypes=dict(NativeStrong=original['NativeStrong'],NativeRegistration=native_read,
        NativeCap2Registration=native_cap2,LayerRegistration=original['BestLayer'],
        TokenRegistration=original['BestToken'],HybridRegistration=original['BestHybrid'],
        TokenOldControl=original['BestToken'])
    for label,cfg in list(prototypes.items()):
        prototypes[label]=dict(cfg,defer_D_request_registration=label not in ('NativeStrong','TokenOldControl'))
    assert all(prototypes[a]['continuous_source_credit_limit']==8
               for a in ('LayerRegistration','TokenRegistration','HybridRegistration','TokenOldControl'))
    assert prototypes['LayerRegistration']['D_running_admission_cap']==2
    assert not prototypes['TokenRegistration'].get('D_prefill_completion_guard')
    source,sha=first['source_capsule'],first['source_capsule_SHA256']
    proof_items={}
    for label,prototype in prototypes.items():
        if not prototype['defer_D_request_registration']:
            continue
        wanted=registration_signature(prototype)
        prior=[p for p in first['qualification'].values() if p['signature']==wanted]
        if prior:
            assert len(prior)==1
            proof_items[label]=prior[0]
            continue
        jobs=[]
        gate='request_registration_q28_'+label+'_gate_v1.json'
        for count,dataset in ((32,'continuous_qualification32_natural.json'),
                               (400,'longbench_balanced400_v1.json')):
            name=f'RegistrationQ28_{label}_Natural{count}_v1'
            cfg=clone(prototype,name,dataset,2.8,2026112201,(4,5),source,sha,correctness=True)
            cfg.pop('final_composition_gate',None)
            save(cfg);jobs.append(name)
        proof_items[label]=dict(jobs=jobs,gate=gate,signature=wanted)
    cells=[];jobs=[]
    for index,seed in enumerate((2026112781,2026112782,2026112783)):
        arms={}
        for label,prototype in prototypes.items():
            name=f'RegistrationConfirm_Main400_q2.8_s{index}_{label}_v1'
            cfg=clone(prototype,name,'longbench_balanced400_v1.json',2.8,seed,(4,5),source,sha)
            cfg.pop('final_composition_gate',None)
            if label in proof_items:
                cfg['registration_deferral_gate']=proof_items[label]['gate']
            cfg.update(registration_intervention=True,scope='The same host-registration fix at frozen primary QPS2.8. '
                'Keep all discovery-frozen cuts,credits,pools and D caps. Seven arms and three new seeds fixed before extension timing. '
                'Do not replace original main or q1.8 repair results. Same known400 prompts,not new-request generalization.')
            save(cfg);arms[label]=name
        ordered=list(arms.values());np.random.default_rng(seed+19).shuffle(ordered);jobs.extend(ordered)
        cells.append(dict(stage='INDEPENDENT_CONFIRMATION',dataset='longbench_balanced400_v1.json',
            QPS=2.8,start=index,seed=seed,GPU_pair=[4,5],jobs=arms))
    write(path,dict(time=time.time(),source_capsule=source,source_capsule_SHA256=sha,
        original_main_plan_SHA256=digest(ROOT/'configs/final_continuous_execution_plan_v1.json'),
        qualification=proof_items,cells=cells,performance_jobs=jobs,
        common_native=dict(LayerRegistration='NativeCap2Registration',
                           TokenRegistration='NativeRegistration',HybridRegistration='NativeRegistration'),
        trigger='The same host-registration gap is a cross-method runtime risk. The first q1.8 registration pilot '
                'has ITLp99=8.822829ms,TPOTp95=8.013300ms,but its same-source old control and independent confirmation are pending. '
                'Validate QPS2.8 irrespective of the remaining q1.8 result,subject to correctness. Do not select a favorable load afterward.',
        evidence='ADAPTIVE_HIGH_LOAD_EXTENSION_FROZEN_BEFORE_EXTENSION_MEASUREMENTS',
        scope='Eight new proof starts plus one exact-source/signature Native proof reused from q1.8. '
              'Twenty-one serving starts after the original controlled bridge drains on4/5. '
              'Different QPS uses original discovery-frozen QPS-specific choices,so cross-QPS differences are not a pure load intervention. '
              'Within each QPS,new and old Token differ only in request-registration deferral. '
              'Both unmodified NativeStrong and exact common-policy Native safety controls remain mandatory.'))
    print('REGISTRATION_Q28_EXTENSION_FROZEN',len(jobs),'serving starts',flush=True)


def run():
    while not (ROOT/'state/controlled_continuous_bridge_complete_v1.json').exists():
        time.sleep(20)
    plan=json.loads((ROOT/'configs'/PLAN).read_text())
    run_plan(plan,'request_registration_q28_extension')
    from .registration_deferral_analysis import main as analyze
    analyze(PLAN,'_q28')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('build','run'));args=parser.parse_args()
    build() if args.action=='build' else run()
