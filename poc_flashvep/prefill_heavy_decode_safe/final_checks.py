"""Record source, test, document, and preservation checks before publication."""
import json,subprocess,time
from .common import *


def main():
    initialize()
    from ..completion_aware_loop_pd.publish import git,preserved
    preserved();commands=[
        ['python','-m','compileall','-q',str(SOURCE.relative_to(REPO))],
        ['python','-m','unittest',*[f'poc_flashvep.prefill_heavy_decode_safe.test_{n}' for n in
            ('gates','envelope_analysis','fixed_trace_packing','record_scope')],'-q']]
    report=REPO/'poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_final_summary.md'
    commands.append(['python','/home/esjung/.codex/skills/asd-ste100/scripts/ste-lint.py',str(report),
        '--disable','synonym-rotation','--json'])
    results=[]
    for command in commands:
        value=subprocess.run(command,cwd=REPO,text=True,capture_output=True)
        results.append(dict(command=command,returncode=value.returncode,stdout=value.stdout,stderr=value.stderr))
        assert value.returncode==0,results[-1]
    files={str(p.relative_to(REPO)):digest(p) for p in SOURCE.rglob('*')
        if p.is_file() and p.suffix in ('.py','.cc') and '__pycache__' not in p.parts}
    write(ROOT/'state/final_checks.json',dict(time=time.time(),passed=True,results=results,
        source_SHA256=files,active_checkout=git('branch','--show-current'),worktree_HEAD=git('rev-parse','HEAD'),
        original_index_preserved=True,external_runtime_patch_preserved=True,
        scope='Syntax, 12 CPU regressions, new-report structural checks, exact source hashes, and preserved user state. These checks do not qualify unrun GPU Serving. Technical launch/start terms denote different events, so synonym rotation is not treated as an error.'))
    print(json.dumps(dict(passed=True,source_files=len(files))))


if __name__=='__main__':main()
