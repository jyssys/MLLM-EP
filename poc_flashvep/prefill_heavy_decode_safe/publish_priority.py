"""Publish only the completed priority comparison through a private index."""
import json,os,subprocess,tempfile,time
from .common import *


def main():
    initialize()
    from ..completion_aware_loop_pd.publish import git,preserved
    preserved();active=git('branch','--show-current');parent=git('rev-parse','refs/heads/'+BRANCH)
    assert parent==BASE,'Inspect an existing publication before adding another one'
    report=REPO/'poc_flashvep/reports/ouro_huginn_loop_token_suffix_128_256_512_quick_comparison.md'
    plan=json.loads((ROOT/'state/priority_width_comparison_plan.json').read_text())
    extra=json.loads((ROOT/'state/priority_outeronly_control_plan.json').read_text())
    names=set(p for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py','.cc') and '__pycache__' not in p.parts)
    names.add(report)
    for name in plan['jobs']+extra['jobs']:
        folder=ROOT/'raw'/name
        assert json.loads((folder/'dispatcher_result.json').read_text())['complete'],name
        names.add(ROOT/'configs'/(name+'.json'))
        names.update(p for p in folder.rglob('*') if p.is_file() and
            not {'worker_source','__pycache__'}&set(p.relative_to(folder).parts) and p.suffix not in ('.pyc','.lock'))
        log=ROOT/'logs'/(name+'.log')
        if log.exists():names.add(log)
    for relative in (
        'analysis/priority_width_comparison_results.json','analysis/priority_outeronly_control_results.json',
        'state/priority_width_comparison_plan.json','state/priority_outeronly_control_plan.json',
        'state/priority_width_report_complete.json','state/priority_width_controller_complete.json',
        'state/user_priority_interruption.json','state/preservation.json','state/private_runtime.patch',
        'configs/frozen_SLO.json','tools/lt_build.json','data/heldout_ouro_L4096.json','data/heldout_huginn_L2048.json'):
        names.add(ROOT/relative)
    paths=sorted(str(p.relative_to(REPO)) for p in names)
    assert all((REPO/p).is_file() and (REPO/p).stat().st_size<90_000_000 for p in paths)
    write(ROOT/'state/priority_publication_inventory.json',dict(time=time.time(),files=len(paths),
        original_bytes=sum((REPO/p).stat().st_size for p in paths),
        scope='Completed priority comparison only. Raw source ZIP files remain included. Extracted worker_source directories and bytecode are redundant and excluded. The broader discovery campaign remains active.'))
    paths.append(str((ROOT/'state/priority_publication_inventory.json').relative_to(REPO)))
    with tempfile.TemporaryDirectory(prefix='priority_width_publication_') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'));git('read-tree',parent,env=env)
        for offset in range(0,len(paths),100):git('add','-f','--',*paths[offset:offset+100],env=env)
        changed=git('diff','--cached','--name-only',parent,env=env).splitlines()
        assert changed and set(changed)<=set(paths)
        code=[p for p in changed if p.endswith(('.py','.cc'))]
        if code:git('diff','--cached','--check',parent,'--',*code,env=env)
        tree=git('write-tree',env=env)
        commit=git('commit-tree',tree,'-p',parent,'-m','Measure matched Loop and 128/256/512-token continuation costs',env=env)
        write(ROOT/'state/priority_publication_intent.json',dict(branch=BRANCH,parent=parent,commit=commit,time=time.time()))
        git('update-ref','refs/heads/'+BRANCH,commit,parent)
        git('push','origin',commit+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==commit
        receipt=ROOT/'state/priority_publication_receipt.json'
        write(receipt,dict(time=time.time(),branch=BRANCH,evidence_commit=commit,remote_SHA=remote,
            verified=True,files=len(paths),scope='Priority comparison complete. Workload discovery and conditional migration remain in progress.'))
        git('add','-f','--',str(receipt.relative_to(REPO)),env=env)
        tree=git('write-tree',env=env)
        final=git('commit-tree',tree,'-p',commit,'-m','Record verified priority width comparison publication',env=env)
        git('update-ref','refs/heads/'+BRANCH,final,commit)
        git('push','origin',final+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==final
    preserved();assert git('branch','--show-current')==active
    write(ROOT/'state/priority_remote_verification.json',dict(time=time.time(),branch=BRANCH,commit=final,remote_SHA=remote,verified=True))
    print(json.dumps(dict(branch=BRANCH,commit=final,remote_SHA=remote,files=len(paths))))


if __name__=='__main__':main()
