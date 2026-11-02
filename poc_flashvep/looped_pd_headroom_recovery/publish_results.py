"""Create a real results-branch commit without changing the user's active index."""
import argparse
import subprocess
import tempfile
from .common import *

PUBLICATION_BRANCH=BRANCH+'-results'
FINAL=REPO/'poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom_final_summary.md'
SPEC=REPO/'poc_flashvep/reports/looped_pd_kv_admission_profiling_adaptive_prefill_headroom_poc_spec.md'


def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()


def push(head):
    command=['git','-c','http.version=HTTP/1.1','-c','http.postBuffer=1073741824','push','origin',
        f'{head}:refs/heads/{PUBLICATION_BRANCH}']
    result=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
    query=subprocess.run(['git','ls-remote','origin','refs/heads/'+PUBLICATION_BRANCH],cwd=REPO,capture_output=True,text=True)
    remote=query.stdout.split()[0] if query.returncode==0 and query.stdout.split() else None
    append(ROOT/'state/results_push_attempts.jsonl',dict(time=time.time(),commit=head,returncode=result.returncode,
        remote_SHA=remote,verified=remote==head,stdout=result.stdout[-2000:],stderr=result.stderr[-2000:],
        scope='No force push. A matching remote SHA, not a success-looking message, establishes publication.'))
    if remote!=head:raise RuntimeError('Push is not verified. Preserve the local results branch and inspect results_push_attempts.jsonl.')
    return remote


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--push',action='store_true')
    parser.add_argument('--batch-mib',type=int,default=800)
    a=parser.parse_args()
    assert a.batch_mib>=100,'A batch must hold the largest permitted artifact'
    preserve()
    active=git('branch','--show-current')
    assert active!=PUBLICATION_BRANCH,'Never update the checked-out branch behind its index'
    assert FINAL.exists(),'Do not publish an incomplete campaign as a final result'
    # The working branch and original staged entries remain exactly unchanged.
    # Publication uses a distinct local branch, a temporary index and commit-tree.
    baseline=json.loads((ROOT/'state/preservation.json').read_text())['base']
    probe=subprocess.run(['git','rev-parse','--verify','refs/heads/'+PUBLICATION_BRANCH],cwd=REPO,capture_output=True,text=True)
    parent=probe.stdout.strip() if probe.returncode==0 else baseline
    if a.push and probe.returncode==0:
        # First publish a locally committed batch from an interrupted attempt.
        # Otherwise the next push could combine two bounded batches.
        push(parent)
    archive=ROOT/'archives/manifest.json'
    replacements={ROOT/r['original'] for r in json.loads(archive.read_text())['originals']} if archive.exists() else set()
    files=[p for p in SOURCE.rglob('*.py')]+[SPEC,FINAL]
    files.extend(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
        not p.name.endswith(('.lock','.tmp')) and p.name!='burner.log')
    files=sorted({str(p.relative_to(REPO)) for p in files if p.is_file() and p not in replacements},
        key=lambda f:('/archives/' in f,f))
    assert all((REPO/f).stat().st_size<100_000_000 for f in files),'Archive oversized originals before publication'
    owned=(str(SOURCE.relative_to(REPO))+'/',str(ROOT.relative_to(REPO))+'/',str(SPEC.relative_to(REPO)),str(FINAL.relative_to(REPO)))
    def allowed(path):return any(path.startswith(p) if p.endswith('/') else path==p for p in owned)
    with tempfile.TemporaryDirectory(prefix='headroom_results_index_') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'))
        git('read-tree',parent,env=env)
        batches=[];current=[];size=0
        for name in files:
            length=(REPO/name).stat().st_size
            if current and size+length>a.batch_mib*1024**2:
                batches.append(current);current=[];size=0
            current.append(name);size+=length
        if current:batches.append(current)
        head=parent;exists=probe.returncode==0
        for number,batch in enumerate(batches,1):
            for i in range(0,len(batch),100):git('add','-f','--',*batch[i:i+100],env=env)
            changed=git('diff','--cached','--name-only',head,env=env).splitlines()
            assert all(allowed(p) for p in changed),'Unrelated staged content entered the temporary index'
            if not changed:continue
            prose=[f for f in changed if f.endswith(('.py','.md')) and '/raw/' not in f and f!=str(SPEC.relative_to(REPO))]
            if prose:
                result=subprocess.run(['git','diff','--cached','--check',head,'--',*prose],cwd=REPO,env=env,capture_output=True,text=True)
                if result.returncode:raise RuntimeError(result.stdout+result.stderr)
            tree=git('write-tree',env=env)
            previous=head
            head=git('commit-tree',tree,'-p',previous,'-m',
                f'Record KV/profile headroom evidence, bounded publication batch {number}/{len(batches)}',env=env)
            git('update-ref','refs/heads/'+PUBLICATION_BRANCH,head,previous if exists else '0'*40)
            exists=True
            preserve()
            remote=push(head) if a.push else None
            append(ROOT/'state/results_publication_batches.jsonl',dict(time=time.time(),batch=number,batches=len(batches),
                commit=head,parent=previous,files=len(changed),raw_batch_bytes=sum((REPO/f).stat().st_size for f in batch),
                remote_SHA=remote,verified=remote==head if a.push else False,
                scope='Bounded ordinary fast-forward pushes. The active index remains unchanged. No force push.'))
            print(json.dumps(dict(batch=number,batches=len(batches),commit=head,verified=remote==head if a.push else False)),flush=True)
    assert git('branch','--show-current')==active
    preserve()
    remote=push(head) if a.push else None
    write(ROOT/'state/results_publication.json',dict(time=time.time(),working_branch=active,publication_branch=PUBLICATION_BRANCH,
        commit=head,remote_SHA=remote,push_verified=remote==head if a.push else False,files=len(files),
        original_index_preserved=True,private_runtime_preserved=True,
        scope='The active worktree branch and staged index were not changed. The committed results reside on the separate named results branch.'))
    print(json.dumps(dict(branch=PUBLICATION_BRANCH,commit=head,remote_SHA=remote,push_verified=remote==head if a.push else False)))


if __name__=='__main__':main()
