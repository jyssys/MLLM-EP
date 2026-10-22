"""Bounded phase chaining. No automatic adaptive-policy or quality claims."""
import argparse
import json
import subprocess
import time
from .common import ROOT,REPO,PYTHON,write,append
from .dispatch import completed
from .resources import restore

def all_jobs(name):
    plan=json.loads((ROOT/name).read_text())
    return plan.get('jobs',[j for g in plan.get('groups',[]) for j in g['jobs']])

def run(module,*args):
    child=subprocess.run([PYTHON,'-u','-m',module,*args],cwd=REPO)
    assert child.returncode==0,(module,child.returncode)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--after-p0',action='store_true')
    p.add_argument('--stop-after-p1',action='store_true')
    a=p.parse_args()
    try:
        until=time.monotonic()+4*3600
        while a.after_p0 and not all(completed(j) for j in all_jobs('p0_contention_plan.json')):
            failed=[j['id'] for j in all_jobs('p0_contention_plan.json') if (ROOT/'raw'/j['id']/'failure.json').exists()]
            assert not failed,('P0_FAILURE_REVIEW_REQUIRED',failed)
            assert time.monotonic()<until,'P0 wait exceeded registered bound'
            write(ROOT/'state'/'controller.json',dict(status='WAITING_FOR_LIVE_P0',wall_s=time.time()))
            time.sleep(5)
        # Allow the calibration dispatcher to finish its owned-burner restoration.
        # Starting the next lease before it exits could race with that restoration.
        def p0_dispatch_live():
            from pathlib import Path
            import os
            for proc in Path('/proc').iterdir():
                if not proc.name.isdecimal():
                    continue
                try:
                    words=(proc/'cmdline').read_bytes().decode().split('\0')
                    if proc.stat().st_uid==os.getuid() and 'poc_flashvep.ouro_adaptive_chunk_loop.dispatch' in words:
                        if '--manifest' in words and words[words.index('--manifest')+1].endswith('p0_contention_plan.json'):
                            return True
                except (OSError,UnicodeError):
                    continue
            return False
        while p0_dispatch_live():
            assert time.monotonic()<until,'P0 dispatcher did not finish its lease'
            time.sleep(1)
        run('poc_flashvep.ouro_adaptive_chunk_loop.contention')
        summary=json.loads((ROOT/'contention_summary.json').read_text())
        assert summary['completed_jobs']==summary['registered_jobs']
        append(ROOT/'phase_exit_decisions.jsonl',dict(phase=0,status='CONDITIONAL' if summary['triggered_pairs'] else 'PASS',
            wall_s=time.time(),reason='Identity and source qualified. Interference cells retain observed drift. Numerical diagnostics may continue. No neutrality or adaptive speedup claim.',summary=summary))
        candidate=json.loads((ROOT/'qualified_runtime_candidate.json').read_text())
        assert candidate['status']=='CPU_QUALIFIED_GPU_NOT_RUN'
        import hashlib
        from pathlib import Path
        actual=subprocess.check_output(['git','diff','--binary'],cwd=candidate['runtime'])
        assert hashlib.sha256(actual).hexdigest()==candidate['patch_sha256'],'Qualified source changed'
        write(ROOT/'active_runtime.json',dict(path=candidate['runtime'],patch_sha256=candidate['patch_sha256'],
            selected_after_P0=True,reason='G3 heterogeneous-boundary CPU regression fixed. GPU numerical qualification follows.'))
        run('poc_flashvep.ouro_adaptive_chunk_loop.dispatch','--manifest',str(ROOT/'p1_diagnostic_plan.json'),'--manage-burners')
        qualification=[]
        for j in all_jobs('p1_diagnostic_plan.json'):
            assert completed(j)
            if j['kind']=='qualify':
                qualification.append(json.loads((ROOT/'raw'/j['id']/'completed.json').read_text())['summary']['isolated_exact_pass'])
        assert qualification and all(qualification),'ISOLATED_NUMERICAL_FAILURE_REVIEW_REQUIRED'
        append(ROOT/'phase_exit_decisions.jsonl',dict(phase=1,status='CONDITIONAL',wall_s=time.time(),
            reason='Isolated numerical tests complete. Batched root cause requires human-readable review. No automatic state/race diagnosis.',
            next='STOP_FOR_ROOT_CAUSE_REVIEW. No adaptive implementation started.'))
        write(ROOT/'state'/'controller.json',dict(status='P1_COMPLETE_REVIEW_REQUIRED',wall_s=time.time()))
    except BaseException as error:
        write(ROOT/'state'/'controller_failure.json',dict(error=repr(error),wall_s=time.time()))
        raise
    finally:
        restore(list(range(8)))

if __name__=='__main__':
    main()
