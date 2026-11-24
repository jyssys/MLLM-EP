"""Measure frozen-trace Main400 one-factor policies after correctness gates."""
import argparse
import fcntl
import json
import subprocess
import time
import traceback
from .common import ROOT,V2,REPO,PYTHON,write,digest,preserve,resources
from .runtime import bind,copy_data,dispatch_allowed

PLAN='main400_native_relative_policy_confirmation_v2.json'


def prepare():
    runner=bind();qualification=json.loads((ROOT/'configs/root_cause_policy_qualification_v1.json').read_text())
    items={row['label']:row for row in qualification['items']};blocks=[];jobs=[]
    addendum=json.loads((ROOT/'configs/native_common0_reference_addendum_v1.json').read_text())
    items['NativeCommon0']=addendum['item']
    labels=('NativeStrong','NativeCommon0','NativeCommon2','StockToken','StockCap2','StockCap2RegOpen','NativeRegOpen',
        'Layer','LayerRegOpen','LayerRetirement','Hybrid','HybridRegOpen','HybridRetirement')
    for start in range(3):
        old=V2/'configs'/f'V2_C3_Main400Natural_q2.0_s{start}_NativeStrong.json'
        frozen=json.loads(old.read_text());arms={};primers={}
        if start==0:order=list(labels)
        elif start==1:order=list(reversed(labels))
        else:
            import numpy as np
            order=np.random.default_rng(2026112403).permutation(labels).tolist()
        for label in order:
            item=items[label];cfg=json.loads((ROOT/'configs'/(item['steps']['Natural400']+'.json')).read_text())
            name=f'NS_Main400Natural_s{start}_{label}'
            cfg.update(name=name,gpus=[0,1],seed=frozen['seed'],arrival_offsets=frozen['arrival_offsets'],
                qps=2.0,serial=False,ignore_eos=False,output_cap=2048,
                correctness_only=False,state_capture=False,axis_state_capture=False,gpu_state_digest=False,
                qualification_gate_v2=item['gate'],scope='Fresh Main400 Natural C3 policy intervention. '
                    'Preserve the historical prompt order and exact scheduled Poisson offsets. '
                    'Do not pool these starts with historical or heldout results.')
            assert cfg['dataset_SHA256']==frozen['dataset_SHA256'] and cfg['count']==400
            assert runner.signature(cfg)==item['signature'],label
            runner.save(cfg);arms[label]=name;jobs.append(name)
            primer_name=f'NS_Main400Prime8_s{start}_{label}'
            control=json.loads((ROOT/'configs'/(item['steps']['Prime8']+'.json')).read_text())
            primer=dict(cfg,name=primer_name,dataset=control['dataset'],dataset_SHA256=control['dataset_SHA256'],count=8,
                serial=True,arrival_offsets=[0.]*8,ignore_eos=True,output_cap=16,
                correctness_only=True,state_capture=not bool(cfg.get('axis_kind')),axis_state_capture=bool(cfg.get('axis_kind')),
                gpu_state_digest=True,axis_capture_hidden_last_only=True,
                official_compiler_cache_regime='OFFICIAL_CACHE_PRIMING_ONLY',
                scope='Correctness-only warm priming under the actual continuous source/admission policy and timed-start model seed. '
                    'This is not an independent performance start.')
            from ..dense_ar_output_aware_v2.work_contract import regenerate
            regenerate(primer,json.loads((ROOT/'data'/primer['dataset']).read_text()))
            runner.save(primer);primers[label]=primer_name
        blocks.append(dict(start=start,seed=frozen['seed'],pair=[0,1],arms=arms,primers=primers,order=order,
            historical_trace_config=str(old.relative_to(REPO)),historical_trace_config_SHA256=digest(old)))
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    contrasts=qualification['contrasts']+[dict(candidate='NativeCommon0',control='NativeCommon2',
        changed_signature_fields=addendum['changed_signature_fields'])]
    write(path,dict(time=time.time(),blocks=blocks,contrasts=contrasts,ordered_jobs=jobs,
        scope='Single physical0/1 pair for all timing arms. Other pairs run only approved burners. '
            'Qualification can use four pairs concurrently but is not performance evidence. '
            'Wait for every qualification controller and serial CPC controller to finish before primary timing. '
            'This conservative schedule excludes concurrent research-pair host/storage/NVLink interference without assuming it is below5%. '
            'Agg remains the separately reproduced historical Main400 reference,not a fabricated fresh arm. '
            'Forward,reverse and fixed-shuffle orders are frozen before measurement. '
            'Keep original full-compiled Stock and both exact Common0/Common2 Native references. One-factor controls are explicitly distinct from best-of-family baselines. '
            'Different natural output work and every decode failure remain measured outcomes. '
            'No new QPS sweep,partition search or dynamic active-decode insertion.'))
    log=ROOT/'logs/main400_native_relative_confirmation_v2.log'
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_native_ttft_slo.serving_confirmation','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/main400_native_relative_controller_v2.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    bind()
    from ..dense_ar_p_critical_path.stable_source import install
    from ..dense_ar_p_critical_path.warm_serving_followup import expected_roots,check_reuse
    from ..dense_ar_p_critical_path.quality import compare as quality
    from ..dense_ar_output_aware_v2.derived_metrics import summarize
    from .qualification import state_audit
    runner=install();plan=json.loads((ROOT/'jobs'/PLAN).read_text());results=[];owner=resources()
    while True:
        ready=all((ROOT/'state'/f'policy_qualification_lane{lane}_complete_v1.json').exists() for lane in range(4))
        ready=ready and (ROOT/'state/serial_native_relative_profile_complete_v2.json').exists()
        if ready or not dispatch_allowed():break
        time.sleep(10)
    required=('NativeStrong','NativeCommon2','StockToken','Layer','Hybrid')
    base_ok=all((ROOT/'state'/('ns_'+label.lower()+'_composition_v1.json')).exists() and
        json.loads((ROOT/'state'/('ns_'+label.lower()+'_composition_v1.json')).read_text())['passed'] for label in required)
    with (ROOT/'state/pair_0_1.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for block in plan['blocks']:
                for label in block['order']:
                    name=block['arms'][label];cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
                    gate=ROOT/'state'/cfg['qualification_gate_v2']
                    if not base_ok or not gate.exists() or not json.loads(gate.read_text())['passed'] or not dispatch_allowed():
                        results.append(dict(name=name,label=label,start=block['start'],status='NOT_RUN',reason='CORRECTNESS_GATE_OR_BUDGET'));continue
                    value=dict(name=name,label=label,start=block['start'],status='FAILED')
                    try:
                        primer=block['primers'][label]
                        assert not (ROOT/'raw'/primer).exists() and not (ROOT/'raw'/name).exists(),'Preserve every live or completed execution'
                        owner.stop([0,1]);assert runner.execute(primer),primer;state=state_audit(primer);assert state['passed']
                        roots=expected_roots(primer)
                        assert set(roots)=={'prefill','decode'}
                        owner.stop([0,1]);assert runner.execute(name),name
                        reuse=check_reuse(name,roots);metric=summarize(name);score=quality(name)
                        value.update(status='COMPLETE',quality_passed=score['passed'],cache_reuse=reuse,
                            primer_state=state,metrics={key:metric[key] for key in ('TTFT_p95_ms','ITL_p99_ms','TPOT_p95_ms',
                                'output_tokens','Joint_SLO_attainment','SLO_qualified_output_tokens_per_second')})
                        if not score['passed']:value['evidence']='FAILED_QUALITY_GATE'
                    except Exception as error:value.update(error=repr(error),traceback=traceback.format_exc())
                    results.append(value)
                    write(ROOT/'state/main400_native_relative_progress_v2.json',dict(time=time.time(),results=results))
                    print('NS_MAIN400_CONFIRMATION',json.dumps({key:val for key,val in value.items() if key not in ('cache_reuse','primer_state','traceback')}),flush=True)
                    # Stop unsafe/invalid arms. Do not retry them or remove their evidence.
                    if value['status']!='COMPLETE' or not value.get('quality_passed',False):
                        for future in plan['blocks']:
                            if future['start']>block['start']:
                                future['order']=[arm for arm in future['order'] if arm!=label]
        finally:owner.restore([0,1]);preserve()
    write(ROOT/'state/main400_native_relative_complete_v2.json',dict(time=time.time(),complete=True,results=results,
        scope='The immutable plan retains all proposed cells. Missing cells are NOT_RUN,not silently removed. '
            'A completed run with a failed quality gate is not a qualified performance result. Decode gates are calculated independently.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'));args=parser.parse_args()
    print(prepare() if args.action=='prepare' else run(),flush=True)
