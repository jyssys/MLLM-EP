"""Reuse the scoped burner guard and its pending-restore protection."""
import sys
import zipfile
import os
import time
from types import SimpleNamespace
from poc_async_recurrent_pp.burner_scope import pending_burner
from poc_async_recurrent_pp.contract import *
from poc_looped_multigpu_kv_placement.scripts import gpu_session as shared
from poc_ouro_reasoning_fidelity_multigpu import guard_cleanup


def main():
    shared.ROOT, shared.PROJECT, shared.ASSETS = ROOT, PROJECT, ASSETS
    guard_cleanup.ASSETS = ASSETS
    requested = list(map(int, sys.argv[sys.argv.index('--gpus')+1].split(',')))
    assert set(requested) <= set(GPUS)
    # A previous scoped restore can still import vLLM on the CPU.
    # Wait for that tracked launcher before the shared guard inventories workers.
    for status in (ROOT/'outputs').glob('*/guard_status.json'):
        for row in read(status).get('restored_burner_workers',[]):
            devices=row['physical_gpu']
            devices=devices if isinstance(devices,list) else [devices]
            relevant=set(devices)&set(requested)
            if not relevant:
                continue
            deadline=time.monotonic()+180
            while True:
                process=Path('/proc')/str(row['pid'])
                try:
                    if process.stat().st_uid!=os.getuid():
                        break
                    words=[x.decode() for x in (process/'cmdline').read_bytes().split(b'\0') if x]
                except (FileNotFoundError,PermissionError):
                    break
                launcher=('/home/esjung/vllm-ep/run_utilize.sh' in words or
                          ('/home/esjung/vllm-ep/utilize.py' in words and '--gpus' in words))
                if not launcher or all(shared.contexts([gpu]) for gpu in relevant):
                    break
                if time.monotonic()>deadline:
                    raise TimeoutError('A tracked scoped burner launcher did not become ready. No foreign process was signalled.')
                time.sleep(1)
    output = (ROOT/sys.argv[sys.argv.index('--output')+1]).resolve()
    assert output.is_relative_to(ROOT/'outputs') and not output.exists()
    expected = sys.argv[sys.argv.index('--')+1:]
    groups = []
    prior_write=shared.write
    def snapshot(path,value):
        if path.name=='run_contract.json':
            value['cuda_allocator']='expandable_segments:True'
        prior_write(path,value)
        if path.name=='run_contract.json':
            with zipfile.ZipFile(path.parent/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
                for source in sorted(ROOT.rglob('*.py')):
                    archive.write(source,arcname=str(source.relative_to(ROOT)))
    shared.write=snapshot
    original = shared.subprocess.Popen
    registered=guard_cleanup.registered_launch(original, expected, output, groups)
    def launch(command,*args,**kwargs):
        if list(command)==expected and kwargs.get('start_new_session') is True:
            kwargs['env']=dict(kwargs['env'],PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        elif '/home/esjung/vllm-ep/utilize.py' in command and '--gpu' in command:
            gpu=int(command[command.index('--gpu')+1])
            existing=pending_burner([gpu])
            if existing is not None:
                return SimpleNamespace(pid=existing)
        elif '/home/esjung/vllm-ep/run_utilize.sh' in command:
            devices=list(map(int,kwargs['env']['VLLM_UTILIZE_GPUS'].split(',')))
            existing=pending_burner(devices)
            if existing is not None:
                return SimpleNamespace(pid=existing)
        return registered(command,*args,**kwargs)
    shared.subprocess.Popen = launch
    try:
        shared.main()
    finally:
        shared.subprocess.Popen = original
        if output.exists():
            guard_cleanup.finish(groups, requested, output, shared.contexts)


if __name__ == '__main__':
    main()
