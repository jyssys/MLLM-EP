"""Qualify the same port/cache policy on every prospective serving arm."""
import fcntl
import json
from pathlib import Path
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind,resources,preserve,dispatch_allowed
from .cache_port_audit import build,freeze_ports

PLAN='common_fixed_ports_qualification_v6.json'
LANES={1:['StockToken','Hybrid','HybridIso8Approx','HybridGCControl','HybridCommon16','HybridIsoCommon16'],
    3:['NativeStrong','NativeCommon2','Layer','Layer8','NativeCommon16','StockCommon16']}


def gate(label):return 'pc_fixed_port_'+label.lower()+'_composition_v6.json'


def native_state(name):
    from ..dense_ar_longbench400_pd.decode_cap_discovery import full_native_state
    return full_native_state(name,False)


def parent(label,count):
    if label=='NativeCommon16':return f'PC_CommonPolicy_{label}_Natural{count}_StateCaptureFixV3'
    if label in ('NativeStrong','NativeCommon2'):
        return f'PC_CoordQual_{label}_Natural{count}_StateCaptureFixV3'
    prefix='PC_CommonPolicy_' if label in ('StockCommon16','HybridCommon16','HybridIsoCommon16') else 'PC_CoordQual_'
    return prefix+label+f'_Natural{count}'


def stamp(cfg,name,pair,ports):
    source=build(cfg['source_capsule'])
    cfg.update(name=name,gpus=pair,source_capsule=source.name,source_capsule_SHA256=digest(source),
        fixed_NIXL_control_ports=ports,stable_worker_source_path=True)
    return cfg


def prepare():
    runner=bind();ports=freeze_ports();items=[]
    for lane,labels in LANES.items():
        pair=[2*lane,2*lane+1]
        for label in labels:
            jobs=[]
            for count in (32,400):
                cfg=json.loads((ROOT/'configs'/(parent(label,count)+'.json')).read_text())
                name=f'PC_PortQual_{label}_Natural{count}';stamp(cfg,name,pair,ports);runner.save(cfg);jobs.append(name)
            final=cfg;strict=[];control=None
            if label in ('Layer','Hybrid','Layer8','HybridIso8Approx'):
                for suffix in ('NormalGC','DeferredGC'):
                    cfg=json.loads((ROOT/'configs'/f'PC_CoordQual_{label}_{suffix}_Strict8.json').read_text())
                    name=f'PC_PortQual_{label}_{suffix}_Strict8';stamp(cfg,name,pair,ports);runner.save(cfg);strict.append(name)
            elif label!='StockToken':
                cfg=json.loads((ROOT/'configs'/(jobs[0]+'.json')).read_text())
                dataset='compiler_control8_original_v1.json';data=json.loads((ROOT/'data'/dataset).read_text())
                control=f'PC_PortQual_{label}_State8'
                cfg.update(name=control,dataset=dataset,dataset_SHA256=digest(ROOT/'data'/dataset),count=8,
                    serial=True,ignore_eos=True,output_cap=16,arrival_offsets=[0.]*8,
                    scope='Fresh8-request state/coverage/lifetime control. No natural-quality or performance claim.')
                from ..dense_ar_output_aware_v2.work_contract import regenerate
                regenerate(cfg,data);runner.save(cfg)
            items.append(dict(label=label,lane=lane,pair=pair,jobs=jobs,strict=strict,control=control,
                gate=gate(label),signature=runner.signature(final),source_capsule_SHA256=final['source_capsule_SHA256']))
    path=ROOT/'configs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),items=items,ports=ports,
        scope='Fresh qualification for all12 predeclared Native,axis and common-policy controls. '
            'All use the same frozen address policy and verified stable worker paths. Worker bytes remain identical to each parent. '
            'Stock needs the exact once-only cache/strict triple. Each other arm gets fresh8-state or exact8,then Natural32/400. '
            'Use the unchanged compiled-C1 quality contract. A failed strong baseline blocks every new performance comparison. '
            'Do not override compiler keys,copy cached outputs or change math,precision,arrival,Dataset,SLO or static actions.'))
    children=[]
    for lane in (1,3):
        log=ROOT/'logs'/f'common_fixed_ports_qualification_lane{lane}_v6.log';assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.port_qualification','run',str(lane)],
                cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(lane=lane,pid=child.pid))
    write(ROOT/'state/common_fixed_ports_qualification_controllers_v6.json',dict(time=time.time(),children=children));return children


def run(lane):
    from .stable_source import install
    runner=install()
    from .coordinator_state_fix import compare as strict_compare
    from ..dense_ar_longbench400_pd.qualification import audit as coverage
    from .quality import compare as quality
    plan=json.loads((ROOT/'configs'/PLAN).read_text());items=[row for row in plan['items'] if row['lane']==lane]
    pair=[2*lane,2*lane+1];owner=resources();results=[];baseline=ROOT/'state/fixed_port_stock_strict_complete_v6_reader_fix1.json'
    def check(name):
        cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        state=coverage(name) if cfg.get('axis_kind') else native_state(name)
        assert state['passed'],name
        if cfg.get('axis_kind'):
            assert state['decode_full36_observation_complete'] and all(row['status']=='PASS' for row in state['imported_prefix_KV'])
        return state
    with (ROOT/'state'/f'pair_{pair[0]}_{pair[1]}.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for index,item in enumerate(items):
                proof=dict(time=time.time(),passed=False,signature=item['signature'],source_capsule_SHA256=item['source_capsule_SHA256'],
                    stages=[],strict=None,evidence='CORRECTNESS_ONLY')
                try:
                    if lane==1 or index>0:
                        while not baseline.exists():
                            assert dispatch_allowed(),'Dispatch budget expired before the required strict baseline'
                            time.sleep(5)
                        assert json.loads(baseline.read_text())['passed'],'Required Stock cache/strict counterfactual failed'
                    if item['strict']:
                        for name in item['strict']:owner.stop(pair);assert runner.execute(name),name;check(name)
                        proof['strict']=strict_compare(*item['strict']);assert proof['strict']['passed'],proof['strict']
                    elif item['control']:
                        name=item['control'];owner.stop(pair);assert runner.execute(name),name
                        proof['state8']=dict(name=name,state=check(name))
                    else:
                        proof['strict']=json.loads(baseline.read_text());assert proof['strict']['passed']
                    for name in item['jobs']:
                        owner.stop(pair);assert runner.execute(name),name;state=check(name);score=quality(name)
                        assert score['passed'],score
                        proof['stages'].append(dict(name=name,state=state,quality=score))
                    proof.update(passed=True,full400_passed=True,quality_passed=True)
                except Exception as error:proof.update(error=repr(error),evidence='FAILED')
                write(ROOT/'state'/item['gate'],proof)
                results.append(dict(label=item['label'],passed=proof['passed'],error=proof.get('error')))
                write(ROOT/'state'/f'common_fixed_ports_qualification_lane{lane}_progress_v6.json',dict(time=time.time(),results=results))
                print('FIXED_PORT_COMPOSITION',results[-1],flush=True)
        finally:owner.restore(pair);preserve()
    write(ROOT/'state'/f'common_fixed_ports_qualification_lane{lane}_complete_v6.json',dict(time=time.time(),results=results,complete=True))


def recheck_completed_native8():
    """Recover a CPU reader result. Never repeat a GPU run or promote its composition."""
    bind();preserve();name='PC_PortQual_NativeStrong_State8'
    receipt=ROOT/'raw'/name/'dispatcher_result.json'
    assert json.loads(receipt.read_text())['complete']
    original=ROOT/'state'/gate('NativeStrong')
    assert 'missing 1 required positional argument' in json.loads(original.read_text())['error']
    result=native_state(name)
    target=ROOT/'state/fixed_port_native8_CPU_reader_recheck_v6.json'
    assert not target.exists()
    write(target,dict(time=time.time(),passed=result['passed'],name=name,state=result,
        original_composition_proof_SHA256=digest(original),GPU_rerun=False,
        composition_still_failed=True,
        scope='The completed GPU state control is read again with the required Native/Deflection argument. '
            'Preserve the original TypeError and failed composition. Natural32/400 and performance remain NOT_RUN. '
            'The independent strong Stock strict gate failed. This reader recovery cannot bypass it.'))
    return result['passed']


if __name__=='__main__':
    import sys
    if sys.argv[1]=='recheck-native8':print(recheck_completed_native8(),flush=True)
    else:print(prepare() if sys.argv[1]=='prepare' else run(int(sys.argv[2])),flush=True)
