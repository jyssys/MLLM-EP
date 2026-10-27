"""Save CPU test evidence without changing active GPU workers."""
import subprocess
import sys
import time
from .common import REPO,ROOT,write,digest,EP_PYTHON

def main():
    commands=[
        [sys.executable,'-m','unittest','poc_flashvep.refinement_ep_allocation.test_contract','-v'],
        [sys.executable,'-m','unittest','poc_flashvep.refinement_ep_allocation.test_replay','-v'],
        [EP_PYTHON,'-m','unittest','poc_flashvep.refinement_ep_allocation.test_semantic','-v'],
        [sys.executable,'-m','compileall','-q','poc_flashvep/refinement_ep_allocation'],
        ['git','diff','--check','--','poc_flashvep/refinement_ep_allocation',
         'poc_flashvep/reports/00_runtime_audit.md','poc_flashvep/reports/01_baseline_reproduction.md',
         'poc_flashvep/reports/11_final_decision.md']]
    evidence=[]
    for command in commands:
        start=time.time()
        result=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
        evidence.append(dict(command=command,start=start,end=time.time(),returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr,evidence_kind='CPU_TEST_NOT_GPU_MEASUREMENT'))
    write(ROOT/'cpu_validation.json',dict(tests=evidence,
        test_source_sha256=digest(REPO/'poc_flashvep/refinement_ep_allocation/test_contract.py'),
        all_pass=all(r['returncode']==0 for r in evidence)))
    for r in evidence:
        print(r['returncode'],' '.join(r['command']))
    if not all(r['returncode']==0 for r in evidence):
        raise SystemExit(1)

if __name__=='__main__':
    main()
