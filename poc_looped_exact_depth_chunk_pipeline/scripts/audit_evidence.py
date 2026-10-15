"""Snapshot reproducibility facts without changing shared runtimes."""
import hashlib
import subprocess
import sys
import importlib.metadata
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    initial=read(ROOT/'results/environment.json')
    base=Path('/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages')
    paths=[base/'vllm/model_executor/models/ouro.py',base/'vllm/v1/worker/gpu/model_runner.py',
           base/'vllm/v1/attention/backends/flash_attn.py',MODEL/'modeling_ouro.py',MODEL/'configuration_ouro.py']
    source_manifest=ROOT/'results/runtime_sources.json'
    current=[dict(path=str(path),sha256=sha(path)) for path in paths]
    if source_manifest.exists():
        runtime_unchanged=read(source_manifest)==current
    else:
        write(source_manifest,current)
        runtime_unchanged=True
    original=Path('/home/esjung/MLLM-EP-github')
    hashes={k:hashlib.sha256(subprocess.check_output(cmd,cwd=original)).hexdigest() for k,cmd in
        [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]}
    checks=dict(authoritative_spec_unchanged=sha(SPEC)==sha(ROOT/'authoritative_spec.md')==initial['spec_sha256'],
        original_dirty_and_staged_unchanged=hashes==initial['original_diffs'],
        checkpoint_source_files_unchanged=all(sha(MODEL/name)==value for name,value in initial['model_files'].items()),
        runtime_sources_unchanged=runtime_unchanged,
        only_permitted_masks=True)
    for path in (ROOT/'results').rglob('run_contract.json'):
        assert read(path)['mask'] in ['4','4,5','4,5,6,7'],path
    sys.path.insert(0,str(ASSETS/'cpu_deps'))
    versions={p:importlib.metadata.version(p) for p in ['math-verify','latex2sympy2_extended','sympy','pyarrow']}
    write(ROOT/'results/audit.json',dict(checks=checks,all_pass=all(checks.values()),evaluator_dependencies=versions,
        tolerance=TOLERANCE,scope='Read-only source, GPU-mask, and preservation audit. Scientific gates remain separate.'))
    assert all(checks.values())
    print(checks)


if __name__=='__main__':main()
