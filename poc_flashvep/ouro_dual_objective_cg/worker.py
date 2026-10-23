"""Reuse the validated model loader and numerical probes in a new attempt root."""
from pathlib import Path
from .common import ROOT, FORK, guard, write, digest

def main():
    guard()  # Reject forbidden devices before any Torch CUDA call.
    import hashlib,json,subprocess
    lock=json.loads((ROOT/'model_source_locks.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=FORK,text=True).strip()==lock['pin']
    assert hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=FORK)).hexdigest()==lock['diff_sha256']
    from ..ouro_budget_joint_cg import worker as inherited
    from ..ouro_budget_joint_cg import benchmark
    inherited.ROOT=ROOT;inherited.guard=guard
    benchmark.ROOT=ROOT;benchmark.guard=guard
    original_snapshot=inherited.source_snapshot
    def snapshot(target):
        parent=original_snapshot(target)
        files={p.name:dict(sha256=digest(p),text=p.read_text()) for p in Path(__file__).parent.glob('*.py')}
        write(target/'dual_sources.json',files)
        return dict(parent,dual_harness_sha256=digest(target/'dual_sources.json'))
    inherited.source_snapshot=snapshot
    original_run=benchmark.run_job
    def run(model,job,target):
        if job['kind'] in ('dual_serving','dual_quality'):
            from .serving_eval import run_job
            return run_job(model,job,target)
        return original_run(model,job,target)
    benchmark.run_job=run
    inherited.main()

if __name__=='__main__': main()
