"""Check source, tests, and pre-existing user state before publication."""
import subprocess,sys
from .common import *


def main():
    initialize()
    from ..completion_aware_loop_pd.publish import preserved,git
    preserved()
    from ..huginn_net_benefit import closure as audit
    audit.ROOT=ROOT;audit.BASE=BASE;audit.SOURCE=SOURCE
    audit.inherited_source_audit()
    commands=[
        [sys.executable,'-m','unittest','poc_flashvep.loop_width_gemm_validation.test_contracts','-v'],
        [sys.executable,'-m','unittest','poc_flashvep.idle_intervention_dynamic_loop.test_protocol','-v'],
        [sys.executable,'-m','compileall','-q',str(SOURCE)]]
    results=[]
    for command in commands:
        result=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
        results.append(dict(command=command,returncode=result.returncode,output=result.stdout+result.stderr))
    write(ROOT/'state/final_source_verification.json',dict(time=time.time(),checks=results,
        all_passed=all(r['returncode']==0 for r in results),active_branch=git('branch','--show-current'),
        preservation_passed=True,source_hashes={str(p.relative_to(REPO)):digest(p) for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py','.cc')},
        scope='Final source inventory. Per-run frozen snapshots remain authoritative for earlier measurements.'))
    assert all(r['returncode']==0 for r in results)
    print('SOURCE_AND_PRESERVATION_CHECKS_PASS')


if __name__=='__main__':main()
