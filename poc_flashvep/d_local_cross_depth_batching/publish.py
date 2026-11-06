"""Publish only new campaign files without changing the user's Git index."""
import argparse
import tempfile
import time
from .common import *

FINAL=REPO/'poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_final_summary.md'
SPEC=FINAL.with_name('ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_poc_spec.md')


def main(action,push=False):
    initialize()
    from ..looped_pd_autonomous_headroom import publish_campaign as inherited
    inherited.ROOT,inherited.BRANCH=ROOT,BRANCH
    if action=='archive':
        import shutil
        snapshot=ROOT/'state/burner_restoration_startup.log'
        if (ROOT/'burner.log').exists() and not snapshot.exists():
            shutil.copyfile(ROOT/'burner.log',snapshot)
        inherited.archive();return
    git=inherited.git
    active=git('branch','--show-current');assert active!=BRANCH
    head=git('rev-parse','refs/heads/'+BRANCH)
    assert FINAL.exists() and SPEC.exists() and (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    replaced={ROOT/r['original'] for r in json.loads((ROOT/'archives/manifest.json').read_text())['originals']}
    files=[p for p in SOURCE.rglob('*') if p.is_file() and p.suffix in ('.py','.cc','.md')]
    files += [FINAL,SPEC]
    files += [p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and not p.name.endswith(('.lock','.tmp'))
        and ('.tmp.' not in p.name or 'raw' in p.relative_to(ROOT).parts)
        and p.name not in ('burner.log','final_remote_verification.json')]
    names=sorted({str(p.relative_to(REPO)) for p in files if p not in replaced})
    assert all((REPO/n).stat().st_size<100_000_000 for n in names)
    with tempfile.TemporaryDirectory(prefix='dlocal_cross_depth_publication_') as temporary:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temporary)/'index'))
        git('read-tree',head,env=env)
        batches=[];batch=[];size=0
        for name in names:
            length=(REPO/name).stat().st_size
            if batch and size+length>350*1024**2:
                batches.append(batch);batch=[];size=0
            batch.append(name);size+=length
        if batch:batches.append(batch)
        for number,batch in enumerate(batches,1):
            for index in range(0,len(batch),100):git('add','-f','--',*batch[index:index+100],env=env)
            changed=git('diff','--cached','--name-only',head,env=env).splitlines()
            assert set(changed)<=set(names)
            if not changed:continue
            code=[n for n in changed if n.endswith(('.py','.cc')) and '/raw/' not in n]
            markdown=[n for n in changed if n.endswith('.md') and '/raw/' not in n]
            if code:git('diff','--cached','--check',head,'--',*code,env=env)
            # Preserve the supplied spec's intentional Markdown hard breaks.
            if markdown:git('-c','core.whitespace=-blank-at-eol','diff','--cached','--check',head,'--',*markdown,env=env)
            tree=git('write-tree',env=env);previous=head
            head=git('commit-tree',tree,'-p',previous,'-m',f'Record D-local cross-depth batching PoC ({number}/{len(batches)})',env=env)
            git('update-ref','refs/heads/'+BRANCH,head,previous)
            initialize()
            if push:inherited.push(head)
            print(json.dumps(dict(commit=head,batch=number,batches=len(batches),pushed=push)),flush=True)
        if push:
            remote=inherited.push(head)
            receipt=ROOT/'state/publication_receipt.json'
            write(receipt,dict(time=time.time(),branch=BRANCH,evidence_commit=head,evidence_remote_SHA=remote,
                verified=remote==head,original_index_preserved=True,external_runtime_preserved=True,files=len(names),
                scope='Committed receipt verifies the evidence parent. The final metadata commit has a separate local SHA check.'))
            metadata=[str(receipt.relative_to(REPO)),str((ROOT/'state/publication_attempts.jsonl').relative_to(REPO))]
            git('add','-f','--',*metadata,env=env);tree=git('write-tree',env=env);previous=head
            head=git('commit-tree',tree,'-p',previous,'-m','Record verified D-local publication receipt',env=env)
            git('update-ref','refs/heads/'+BRANCH,head,previous);remote=inherited.push(head)
            write(ROOT/'state/final_remote_verification.json',dict(time=time.time(),branch=BRANCH,
                commit=head,remote_SHA=remote,verified=remote==head))
        initialize();assert git('branch','--show-current')==active
        print(json.dumps(dict(branch=BRANCH,commit=head,pushed=push)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('archive','publish'));parser.add_argument('--push',action='store_true')
    args=parser.parse_args();main(args.action,args.push)
