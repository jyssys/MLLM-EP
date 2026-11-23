"""Repair a Native correctness-observer flag,not the Native model runtime."""
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind

PLAN='coordinator_native_state_capture_fix_v3.json'


def capture_flags(cfg):
    assert cfg['method']=='Native' and not cfg.get('axis_kind')
    return dict(cfg,state_capture=True,axis_state_capture=False)


def prepare():
    runner=bind();items=[]
    for label,parent in (('NativeStrong','PC_CoordQual_NativeStrong'),
        ('NativeCommon2','PC_CoordQual_NativeCommon2'),('NativeCommon16','PC_CommonPolicy_NativeCommon16')):
        jobs=[]
        for count in (32,400):
            source=ROOT/'configs'/(parent+f'_Natural{count}.json');old=json.loads(source.read_text())
            assert not old['state_capture'] and old['axis_state_capture'],'The failed instrumentation contract changed'
            cfg=capture_flags(old);name=parent+f'_Natural{count}_StateCaptureFixV3';cfg['name']=name
            runner.save(cfg);jobs.append(name)
        items.append(dict(label=label,arm='NativeStrong' if label=='NativeStrong' else 'NativeCommon',lane=1,pair=[2,3],
            teacher=None,strict=None,jobs=jobs,gate='pc_coord_'+label.lower()+'_composition_state_capture_v3.json',
            signature=runner.signature(cfg),source_capsule_SHA256=cfg['source_capsule_SHA256']))
    path=ROOT/'configs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),items=items,state_prefix='pc_native_capture_v3',source_capsule_SHA256=None,
        actual_failure='The Native32 GPU run completed. Its state audit could not find native_valid_prefix state files.',
        source_cause='The new builder enabled axis_state_capture for a Native arm. Native uses state_capture instead.',
        fix='Set state_capture=True and axis_state_capture=False only in these new correctness versions.',
        scope='No worker,model,arrival,policy or numerical code changes. Old failed gates and raw runs remain unchanged. '
            'These fresh32/400 captures must pass before the new coordinator Native references can time primary performance. '
            'A missing observer file is not evidence that Native KV state was incorrect.'))
    log=ROOT/'logs/coordinator_native_state_capture_fix_v3.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.native_capture_fix','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/coordinator_native_state_capture_fix_controller_v3.json',dict(time=time.time(),pid=child.pid))
    return child.pid


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':print(prepare(),flush=True)
    else:
        bind()
        from ..dense_ar_output_aware_v2.c3_family import run
        run(1,PLAN)
