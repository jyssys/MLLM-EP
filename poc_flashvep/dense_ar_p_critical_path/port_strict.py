"""Test the identified cache-port factor exactly once before new timing."""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,digest,bind,resources,preserve
from .cache_port_audit import audit,freeze_ports,build

PLAN='fixed_port_stock_strict_v6.json'
COMPLETE='fixed_port_stock_strict_complete_v6.json'


def prepare():
    runner=bind();audit();ports=freeze_ports();jobs=[]
    for suffix in ('NormalGC','DeferredGC','NormalGCRepeat'):
        cfg=json.loads((ROOT/'configs'/f'PC_StableSource_StockToken_{suffix}_Strict8.json').read_text())
        source=build(cfg['source_capsule']);name=f'PC_FixedPort_StockToken_{suffix}_Strict8'
        cfg.update(name=name,source_capsule=source.name,source_capsule_SHA256=digest(source),
            stable_worker_source_path=True,fixed_NIXL_control_ports=ports,gpus=[2,3])
        runner.save(cfg);jobs.append(name)
    path=ROOT/'jobs'/PLAN;assert not path.exists()
    write(path,dict(time=time.time(),jobs=jobs,pair=[2,3],ports=ports,
        scope='Exactly one NormalGC/DeferredGC/NormalGCRepeat triple. Same verified source path and frozen transport addresses. '
            'The prior shared-source trial changed only the recorded NIXL environment port within each role. '
            'Preserve the official compiler hash and loader. Never force a kernel,copy an output or weaken the strict gate. '
            'Require actual cache-root/factor equality and full KV/hidden/greedy equality. '
            'A remaining mismatch rejects this repair as sufficient. Preserve every prior failure. '
            'Any later qualification and timing must apply this transport policy to all Native and axis arms.'))
    log=ROOT/'logs/fixed_port_stock_strict_v6.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.port_strict','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/fixed_port_stock_strict_controller_v6.json',dict(time=time.time(),pid=child.pid));return child.pid


def run(resume=False):
    from .stable_source import install
    runner=install()
    from .coordinator_state_fix import compare
    from ..dense_ar_longbench400_pd.qualification import audit as coverage
    plan=json.loads((ROOT/'jobs'/PLAN).read_text());owner=resources()
    suffix='_reader_fix1' if resume else ''
    result=dict(time=time.time(),passed=False,jobs=[],comparisons=[],cache_factors=[],evidence='CORRECTNESS_ONLY')
    with (ROOT/'state/pair_2_3.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        try:
            for name in plan['jobs']:
                folder=ROOT/'raw'/name;receipt=folder/'dispatcher_result.json'
                reused=receipt.exists()
                if reused:
                    assert resume and json.loads(receipt.read_text())['complete'],'Do not retry a failed GPU run'
                    cfg=ROOT/'configs'/(name+'.json');command=json.loads((folder/'command.json').read_text())
                    assert command['config_SHA256']==digest(cfg)
                    assert json.loads(receipt.read_text())['source_SHA256']==json.loads(cfg.read_text())['source_capsule_SHA256']
                else:
                    assert not folder.exists(),'Inspect a live or interrupted handle instead of restarting it'
                    owner.stop([2,3]);assert runner.execute(name),name
                assert coverage(name)['passed'],name
                folder=ROOT/'raw'/name;ports=json.loads((folder/'control_ports.json').read_text())['ports']
                assert [ports[role] for role in ('prefill','decode')]==plan['ports']['2,3']
                result['jobs'].append(dict(name=name,status='REUSED_COMPLETED_GPU_RUN' if reused else 'COMPLETE'))
                for role in ('prefill','decode'):
                    text=(folder/(role+'.log')).read_text();root=Path(re.search(r'Using cache directory: (.*?) for vLLM',text)[1])
                    path=root/'cache_key_factors.json';factor=json.loads(path.read_text())
                    result['cache_factors'].append(dict(name=name,role=role,cache_root=str(root),metadata_SHA256=digest(path),
                        config_hash=factor['config_hash'],code_hash=factor['code_hash'],compiler_hash=factor['compiler_hash'],
                        port=factor['env']['VLLM_NIXL_SIDE_CHANNEL_PORT']))
                write(ROOT/'state'/('fixed_port_stock_strict_progress_v6'+suffix+'.json'),result)
            result['comparisons']=[compare(plan['jobs'][0],name) for name in plan['jobs'][1:]]
            result['cache_root_equal_by_role']={role:len({row['cache_root'] for row in result['cache_factors'] if row['role']==role})==1
                for role in ('prefill','decode')}
            result['passed']=all(row['passed'] for row in result['comparisons']) and all(result['cache_root_equal_by_role'].values())
        except Exception as error:
            import traceback
            result.update(error=repr(error),traceback=traceback.format_exc(),evidence='FAILED')
        finally:owner.restore([2,3]);preserve()
    target=COMPLETE.replace('.json',suffix+'.json')
    write(ROOT/'state'/target,result);print('FIXED_PORT_STOCK_STRICT',result['passed'],flush=True)


def resume_after_reader_fix():
    preserve();old=ROOT/'state'/COMPLETE;failed=json.loads(old.read_text())
    assert failed['error'].startswith('FileNotFoundError') and not failed['jobs']
    target=ROOT/'state/fixed_port_reader_recovery_v6.json';assert not target.exists()
    plan=json.loads((ROOT/'jobs'/PLAN).read_text());assert (ROOT/'raw'/plan['jobs'][0]/'dispatcher_result.json').exists()
    write(target,dict(time=time.time(),original_failure=failed,original_failure_SHA256=digest(old),
        reason='The CPU coverage reader was imported before the campaign root binding. '
            'The first GPU run already completed successfully. Recheck it without a new GPU call. '
            'Continue only the two immutable configurations with no raw folder. Keep the original failed CPU record.'))
    log=ROOT/'logs/fixed_port_stock_strict_v6_reader_fix1.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.port_strict','run-fix1'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/fixed_port_stock_strict_controller_v6_reader_fix1.json',dict(time=time.time(),pid=child.pid));return child.pid


if __name__=='__main__':
    import sys
    actions={'prepare':prepare,'run':run,'resume':resume_after_reader_fix,'run-fix1':lambda:run(True)}
    print(actions[sys.argv[1]](),flush=True)
