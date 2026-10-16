"""Reuse the validated scoped burner guard without changing previous files."""
from poc_ouro_reasoning_fidelity_multigpu.contract import ROOT, PROJECT, ASSETS
from poc_looped_multigpu_kv_placement.scripts import gpu_session as shared
import zipfile
import os
import sys
import time
import fcntl
os.environ.update(POC4_FAMILY='native',POC4_R='4',POC4_MODEL='ouro')
shared.ROOT, shared.PROJECT, shared.ASSETS = ROOT, PROJECT, ASSETS
original_write=shared.write
def write_with_snapshot(path,value):
    original_write(path,value)
    if path.name=='run_contract.json':
        archive=path.parent/'probe_source.zip'
        assert not archive.exists()
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as bundle:
            for source in sorted(ROOT.rglob('*.py')):
                bundle.write(source,arcname=str(source.relative_to(ROOT)))
        original_write(path.parent/'probe_environment.json',dict(
            inherited_control=dict(POC4_FAMILY='native',POC4_R='4',POC4_MODEL='ouro'),
            reason='Disable any inherited tied-unroll mode in reused observer code.'))
shared.write=write_with_snapshot
if __name__ == '__main__':
    # This campaign has one disjoint-GPU calibration diagnostic. Wait for
    # a verified overlapping job from this campaign, never kill that job.
    requested=list(map(int,sys.argv[sys.argv.index('--gpus')+1].split(',')))
    assert len(requested)==len(set(requested))
    deadline=time.monotonic()+900
    lock_directory=ASSETS/'gpu_locks'
    lock_directory.mkdir(parents=True,exist_ok=True)
    locks=[]
    for gpu in sorted(requested):
        handle=(lock_directory/f'physical_gpu{gpu}.lock').open('a')
        while True:
            try:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                locks.append(handle)
                break
            except BlockingIOError:
                if time.monotonic()>deadline:
                    raise TimeoutError('A scoped GPU lock remains active. No process was signalled.')
                time.sleep(2)
    announced=False
    while True:
        active=[]
        for pid in shared.contexts(requested):
            try:
                process=shared.Path('/proc')/str(pid)
                if process.stat().st_uid!=os.getuid():continue
                fields=(process/'environ').read_bytes().split(b'\0')
                if any(field.startswith(('LOOPED_KV_OUTPUT='+str(ROOT)+'/results/').encode()) for field in fields):
                    active.append(pid)
            except (FileNotFoundError,PermissionError):
                continue
        if not active:break
        if not announced:
            print('Wait for verified overlapping experiment jobs:',active,flush=True)
            announced=True
        if time.monotonic()>deadline:
            raise TimeoutError('An overlapping campaign job did not release its GPU. No process was signalled.')
        time.sleep(2)
    from poc_ouro_reasoning_fidelity_multigpu import guard_cleanup
    output=(ROOT/sys.argv[sys.argv.index('--output')+1]).resolve()
    assert output.is_relative_to(ROOT/'results')
    existed=output.exists()
    assert '--' in sys.argv, 'Use -- before the guarded command'
    expected=sys.argv[sys.argv.index('--')+1:]
    registered=[]
    original_popen=shared.subprocess.Popen
    shared.subprocess.Popen=guard_cleanup.registered_launch(original_popen,expected,output,registered)
    try:
        shared.main()
    finally:
        shared.subprocess.Popen=original_popen
        if not existed and output.exists():
            guard_cleanup.finish(registered,requested,output,shared.contexts)
