"""Qualify cached original compiled Stock only after the exact counterfactual."""
import fcntl
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind,resources,preserve,dispatch_allowed

GATE='pc_coord_stocktoken_stable_source_composition_v5.json'


def prepare():
    runner=bind();jobs=[]
    for count in (32,400):
        parent=ROOT/'configs'/f'PC_CoordQual_StockToken_Natural{count}.json'
        value=json.loads(parent.read_text());name=f'PC_StableSource_StockToken_Natural{count}'
        value.update(name=name,stable_worker_source_path=True);runner.save(value);jobs.append(name)
    path=ROOT/'configs/stable_stock_composition_v5.json';assert not path.exists()
    write(path,dict(time=time.time(),jobs=jobs,signature=runner.signature(value),
        source_capsule_SHA256=value['source_capsule_SHA256'],
        scope='Prospective repair family. First require NormalGC/DeferredGC/NormalGCRepeat exact8 on the shared source path. '
            'Then require new Natural32/400 state,coverage,lifetime and unchanged compiled-C1 quality. '
            'The original Stock model,mode3,graph,attention and numerical flags remain unchanged. '
            'A source-path/cache repair is not a Layer-specific kernel speedup. Preserve all earlier numerical failures.'))
    log=ROOT/'logs/stable_stock_qualification_v5.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.stable_stock_qualification','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/stable_stock_qualification_controller_v5.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    from .stable_source import install
    runner=install();path=ROOT/'state/stable_source_stock_strict_complete_v1.json'
    while not path.exists():
        if not dispatch_allowed():
            write(ROOT/'state'/GATE,dict(time=time.time(),passed=False,evidence='NOT_RUN',reason='DISPATCH_BUDGET'));return
        time.sleep(15)
    strict=json.loads(path.read_text());plan=json.loads((ROOT/'configs/stable_stock_composition_v5.json').read_text())
    result=dict(time=time.time(),passed=False,strict=strict,stages=[],signature=plan['signature'],
        source_capsule_SHA256=plan['source_capsule_SHA256'],evidence='CORRECTNESS_ONLY')
    if not strict['passed']:
        result.update(evidence='FAILED',reason='Exact stable-source counterfactual failed. Do not run performance.')
        write(ROOT/'state'/GATE,result);return
    from ..dense_ar_longbench400_pd.qualification import audit
    from .quality import compare
    owner=resources()
    with (ROOT/'state/pair_2_3.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            for name in plan['jobs']:
                owner.stop([2,3]);assert runner.execute(name),name
                state=audit(name);score=compare(name)
                assert state['passed'] and state['decode_full36_observation_complete'] and score['passed'],name
                assert all(row['status']=='PASS' for row in state['imported_prefix_KV'])
                result['stages'].append(dict(name=name,state=state,quality=score))
            result.update(passed=True,full400_passed=True,quality_passed=True)
        except Exception as error:result.update(error=repr(error),evidence='FAILED')
        finally:owner.restore([2,3]);preserve()
    write(ROOT/'state'/GATE,result)
    print('STABLE_STOCK_COMPOSITION',result['passed'],flush=True)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)
