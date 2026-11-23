"""Test a frozen warm-cache regime once,without erasing cold-cache failures."""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,preserve,resources,dispatch_allowed

PLAN='official_warm_cache_stock_strict_v7.json'
COMPLETE='official_warm_cache_stock_strict_complete_v7.json'


def cache_reuse_observation(log,expected):
    matches=re.findall(r'Using cache directory: (.*?) for vLLM',log)
    assert len(matches)==1 and matches[0]==expected,'Official cache identity changed'
    assert 'Directly load the compiled graph(s)' in log,'No observed official graph reuse'
    assert 'Directly load AOT compilation from path ' in log,'No observed official AOT reuse'
    assert 'Compiling a graph for compile range ' not in log,'A new graph compilation invalidates the warm-only control'
    return dict(cache_root=expected,official_graph_reuse=True,official_AOT_reuse=True,new_graph_compilation=False)


def prepare():
    from .stable_source import install
    runner=install();preserve()
    previous=ROOT/'state/fixed_port_stock_strict_complete_v6_reader_fix1.json'
    cold=json.loads(previous.read_text());assert not cold['passed'] and all(cold['cache_root_equal_by_role'].values())
    diagnostic=ROOT/'analysis/fixed_port_warm_cache_pair_diagnostic.json'
    assert json.loads(diagnostic.read_text())['comparison']['passed']
    roots={row['role']:row['cache_root'] for row in cold['cache_factors']}
    assert all((Path(root)/'cache_key_factors.json').exists() for root in roots.values())
    jobs=[]
    for suffix in ('NormalGC','DeferredGC','NormalGCRepeat'):
        cfg=json.loads((ROOT/'configs'/f'PC_FixedPort_StockToken_{suffix}_Strict8.json').read_text())
        name='PC_OfficialWarm_StockToken_'+suffix+'_Strict8'
        cfg.update(name=name,official_compiler_cache_regime='PREEXISTING_VERIFIED_AOT',
            expected_official_cache_roots=roots)
        runner.save(cfg);jobs.append(name)
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),jobs=jobs,pair=[2,3],expected_cache_roots=roots,
        previous_failed_plan_SHA256=digest(previous),retrospective_warm_pair_SHA256=digest(diagnostic),
        hypotheses=[
            'Cold graph execution and official AOT-loaded execution can differ numerically. Three warm runs should match all8 states.',
            'A changing asynchronous state or request order would still produce state differences within the warm triple.',
            'BI0 nondeterminism not isolated by this regime would also leave warm-repeat differences.'],
        scope='Prospective correctness-only regime test. Execute exactly one new NormalGC/DeferredGC/NormalGCRepeat triple. '
            'Require observed official graph and AOT loading,unchanged cache roots,and exact full KV/hidden/greedy states. '
            'Do not edit a compiler key,inject a compiled artifact,alter model math,or repeat a failed GPU call. '
            'The old cold-control gates remain failed. A warm triple alone does not qualify Natural32/400 or N3 performance.'))
    log=ROOT/'logs/official_warm_cache_stock_strict_v7.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.warm_cache_strict','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/official_warm_cache_stock_strict_controller_v7.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    from .stable_source import install
    runner=install()
    from ..dense_ar_longbench400_pd.qualification import audit
    from .coordinator_state_fix import compare
    plan=json.loads((ROOT/'jobs'/PLAN).read_text());owner=resources()
    result=dict(time=time.time(),passed=False,evidence='CORRECTNESS_ONLY',jobs=[],comparisons=[])
    with (ROOT/'state/pair_2_3.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for name in plan['jobs']:
                assert dispatch_allowed(),'The dispatch budget ended before the next predeclared test'
                assert not (ROOT/'raw'/name).exists(),'Never repeat an existing GPU run'
                owner.stop([2,3]);assert runner.execute(name),name
                assert audit(name)['passed'],name
                observations={role:cache_reuse_observation((ROOT/'raw'/name/(role+'.log')).read_text(),root)
                    for role,root in plan['expected_cache_roots'].items()}
                result['jobs'].append(dict(name=name,status='COMPLETE',cache_reuse=observations))
                write(ROOT/'state/official_warm_cache_stock_strict_progress_v7.json',result)
            result['comparisons']=[compare(plan['jobs'][0],name) for name in plan['jobs'][1:]]
            result['passed']=all(row['passed'] for row in result['comparisons'])
        except Exception as error:
            import traceback
            result.update(error=repr(error),traceback=traceback.format_exc(),evidence='FAILED')
        finally:owner.restore([2,3]);preserve()
    write(ROOT/'state'/COMPLETE,result)
    print('OFFICIAL_WARM_CACHE_STRICT',result['passed'],flush=True)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)
