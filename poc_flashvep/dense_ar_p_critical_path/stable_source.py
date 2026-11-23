"""Keep audited worker bytes at a stable path for compiler cache reuse."""
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
import zipfile
from .common import ROOT,REPO,PYTHON,write,digest,bind,resources,preserve

PLAN='stable_source_stock_strict_counterfactual_v1.json'


def source_tree(archive):
    source_sha=digest(archive);base=ROOT/'references/runtime_sources';base.mkdir(parents=True,exist_ok=True)
    target=base/source_sha
    with (base/(source_sha+'.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with zipfile.ZipFile(archive) as bundle:
            assert all(not name.startswith('/') and '..' not in Path(name).parts for name in bundle.namelist())
            expected={name:hashlib.sha256(bundle.read(name)).hexdigest() for name in bundle.namelist()}
            if not target.exists():
                target.mkdir();bundle.extractall(target)
                write(base/(source_sha+'.json'),dict(source_archive_SHA256=source_sha,files=expected))
            for name,sha in expected.items():assert digest(target/name)==sha,('Stable source was modified',name)
    return target


def install():
    runner=bind()
    if getattr(runner.execute,'_pc_stable_source',False):return runner
    original=inspect.unwrap(runner.execute);source=inspect.getsource(original)
    needle='    env={k:v for k,v in os.environ.items() if not k.startswith'
    assert source.count(needle)==1
    source=source.replace(needle,
        "    if cfg.get('stable_worker_source_path'):\n"
        "        frozen=_pc_stable_source_tree(archive)\n"
        "        write(folder/'worker_source_path.json',dict(path=str(frozen),source_archive_SHA256=digest(archive),\n"
        "            scope='Identical audited bytes at a source-SHA path. Keep the per-job source tree and official compiler hashing.'))\n"
        +needle)
    namespace=dict(original.__globals__);namespace['_pc_stable_source_tree']=source_tree
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    value=namespace['execute'];value._pc_stable_source=True;value._pc_failure_trace=True
    runner.execute=value
    return runner


def prepare():
    runner=bind();parent=json.loads((ROOT/'configs/PC_CoordQual_StockToken_NormalGC_Strict8.json').read_text())
    jobs=[]
    for suffix,deferred in (('NormalGC',False),('DeferredGC',True),('NormalGCRepeat',False)):
        value=dict(parent);name='PC_StableSource_StockToken_'+suffix+'_Strict8'
        value.update(name=name,stable_worker_source_path=True,defer_coordinator_cyclic_gc=deferred,gpus=[2,3])
        runner.save(value);jobs.append(name)
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),jobs=jobs,pair=[2,3],
        observation='Same-source Stock exact8 differed from D KV layer1. Existing caches show a P Reduction R0_BLOCK2048/4096 difference.',
        competing_hypotheses=['Independent compiler autotuning','Different batch/attention metadata','Transfer or state lifetime'],
        intervention='Only reuse a verified source-SHA physical path. Do not override official cache hashes or edit model/compiled code. '
            'Per-job source archives and extracted evidence stay intact. Keep compiled Stock mode3 and graphs.',
        prediction='If source-path-induced independent compilation causes the mismatch,the repeated stable-path runs will use the same artifacts and exact states.',
        falsifier='Different states with shared extracted code and artifact choices reject the compiler-path explanation as sufficient.',
        protocol='NormalGC,DeferredGC,NormalGCRepeat exactly once. No performance run before strict,coverage and fresh Natural32/400 gates. '
            'Do not repeat until a pass. Keep every earlier mismatch and use no tensor/output correction.'))
    log=ROOT/'logs/stable_source_stock_strict_counterfactual_v1.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.stable_source','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/stable_source_stock_strict_controller_v1.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    runner=install();plan=json.loads((ROOT/'jobs'/PLAN).read_text());owner=resources()
    from ..dense_ar_longbench400_pd.qualification import audit
    from .coordinator_state_fix import compare
    result=dict(time=time.time(),passed=False,comparisons=[],evidence='CORRECTNESS_ONLY')
    with (ROOT/'state/pair_2_3.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            for name in plan['jobs']:
                owner.stop([2,3]);assert runner.execute(name),name
                assert audit(name)['passed'],name
            result['comparisons']=[compare(plan['jobs'][0],name) for name in plan['jobs'][1:]]
            result['passed']=all(row['passed'] for row in result['comparisons'])
        except Exception as error:result['error']=repr(error)
        finally:owner.restore([2,3]);preserve()
    write(ROOT/'state/stable_source_stock_strict_complete_v1.json',result)
    print('STABLE_SOURCE_STOCK_STRICT',result['passed'],flush=True)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)
