"""Snapshot source identity without changing the runtime or prior studies."""
import importlib.metadata
import json
import subprocess
import sys
from .common import ROOT,REPO,FORK,MODEL,SPEC,digest,write

def main():
    command=lambda args,cwd=None:subprocess.check_output(args,cwd=cwd,text=True).strip()
    names=['torch','triton','transformers','numpy','flash-attn-4','pytest','matplotlib']
    value=dict(python=sys.version,versions={n:importlib.metadata.version(n) for n in names},
        model_revision=MODEL.name,
        model_files=[dict(name=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(MODEL.iterdir()) if p.is_file() and p.suffix in ('.json','.py','.safetensors','.jinja')],
        runtime_revision=command(['git','rev-parse','HEAD'],FORK),runtime_diff=command(['git','diff','--binary'],FORK),
        runtime_sources=[dict(path=str(p.relative_to(FORK)),sha256=digest(p)) for p in sorted((FORK/'vllm_rlt').rglob('*.py'))],
        runtime_license=command(['git','show','HEAD:LICENSE'],FORK),
        code_sources=[dict(path=str(p.relative_to(REPO)),sha256=digest(p)) for p in sorted((REPO/'poc_flashvep/ouro_adaptive_chunk_loop').glob('*.py'))],
        specification=dict(path=str(SPEC),sha256=digest(SPEC)),
        host=dict(lscpu=command(['lscpu']),cpu_quota={str(p):p.read_text() for p in [
            __import__('pathlib').Path('/sys/fs/cgroup/cpu.max'),
            __import__('pathlib').Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
            __import__('pathlib').Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')] if p.exists()},
            topology=command(['nvidia-smi','topo','-m']),driver=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv'])),
        old_staged_diff_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','diff','--cached','--binary'],cwd=REPO)).hexdigest(),
        paused_PP_program='NOT_RESUMED',scope='SOURCE_LOCK_NOT_PERFORMANCE')
    target=ROOT/'source_locks'/f"lock_{command(['git','rev-parse','--short','HEAD'],REPO)}.json"
    assert not target.exists(),'Source snapshots are immutable'
    write(target,value)
    print('LOCKED',target)

if __name__=='__main__':
    main()
