"""Commit only this namespace with a temporary index. Never force-push."""
import argparse
import subprocess
import tempfile
from .common import *
from .bootstrap import preserve


def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()


def push_verified(head):
    attempts=[['-c','http.version=HTTP/1.1','-c','http.postBuffer=1073741824','push','-u','origin',BRANCH],
        ['push','-u','origin',BRANCH]]
    for arguments in attempts:
        result=subprocess.run(['git',*arguments],cwd=REPO,text=True,capture_output=True)
        probe=subprocess.run(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=REPO,text=True,capture_output=True)
        remote=probe.stdout.split()[0] if probe.returncode==0 and probe.stdout.split() else None
        append(ROOT/'state/publication_push_attempts.jsonl',dict(time=time.time(),commit=head,returncode=result.returncode,
            command=['git',*arguments],stdout=result.stdout[-3000:],stderr=result.stderr[-3000:],remote_SHA=remote,
            verified=remote==head,scope='A successful-looking message is insufficient. Only matching remote SHA establishes publication. No force push.'))
        if remote==head:return remote
    raise RuntimeError('Push did not produce the expected remote SHA. The local commit remains preserved. See publication_push_attempts.jsonl.')


def file_batches(files,maximum_bytes):
    batches=[];batch=[];size=0
    for name in files:
        count=(REPO/name).stat().st_size
        if batch and size+count>maximum_bytes:batches.append(batch);batch=[];size=0
        batch.append(name);size+=count
    if batch:batches.append(batch)
    return batches


def main():
    p=argparse.ArgumentParser();p.add_argument('--push',action='store_true');p.add_argument('--final',action='store_true')
    p.add_argument('--maximum-commit-bytes',type=int,default=700_000_000);a=p.parse_args()
    assert a.maximum_commit_bytes>=100_000_000
    preserve();assert git('branch','--show-current')==BRANCH
    files=list(SOURCE.rglob('*.py'))+[SOURCE/'.gitignore',SPEC]+[REPO/'poc_flashvep/reports'/n for n in REPORTS if (REPO/'poc_flashvep/reports'/n).exists()]
    files.extend((ROOT/'data').glob('*.json'));files.extend((ROOT/'configs').glob('*.json'))
    for folder in (ROOT/'raw').iterdir():
        if not folder.is_dir() or not (folder/'dispatcher_result.json').exists():continue
        files.extend(p for p in folder.rglob('*') if p.is_file())
        log=ROOT/'logs'/(folder.name+'.log')
        if log.exists():files.append(log)
    if a.final:
        files.extend(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.lock'))
    else:
        files.extend(p for p in (ROOT/'state').glob('*.json') if any(v in p.name for v in ('diagnosis','qualification','initial_hardware')))
    files=sorted({str(p.relative_to(REPO)) for p in files if '__pycache__' not in p.parts})
    from .archive import verified_originals
    archived=verified_originals()
    files=[f for f in files if REPO/f not in archived]
    if archived:
        files=sorted(set(files)|{str(p.relative_to(REPO)) for p in (ROOT/'archives').iterdir() if p.is_file()})
    oversized=[f for f in files if (REPO/f).stat().st_size>=100_000_000]
    assert not oversized,('Publish lossless compressed evidence before GitHub push; preserve local originals.',oversized)
    with tempfile.TemporaryDirectory(prefix='atomic_pd_publish_') as directory:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(directory)/'index'));git('read-tree','HEAD',env=env)
        for i in range(0,len(files),100):git('add','-f','--',*files[i:i+100],env=env)
        staged=git('diff','--cached','--name-only',env=env).splitlines()
        assert all(any(f.startswith(p) if p.endswith('/') else f==p for p in OWNED) for f in staged)
        prose=[f for f in staged if (f.endswith('.py') or f.endswith('.md')) and '/raw/' not in f]
        if prose:
            check=subprocess.run(['git','diff','--cached','--check','--',*prose],cwd=REPO,env=env,text=True,capture_output=True)
            if check.returncode:print(check.stdout+check.stderr,flush=True)
            check.check_returncode()
        batches=file_batches(staged,a.maximum_commit_bytes)
        # Keep each upload below GitHub's per-push pack limit. This temporary
        # index contains only owned changes. The ordinary index is untouched.
        if len(batches)>1:
            for i in range(0,len(staged),100):git('reset','-q','HEAD','--',*staged[i:i+100],env=env)
        for index,batch in enumerate(batches):
            if len(batches)>1:
                for i in range(0,len(batch),100):git('add','-f','--',*batch[i:i+100],env=env)
            message='Record Atomic Prefill PD '+('final measured study' if a.final and index==len(batches)-1 else 'implementation and qualification checkpoint')
            if len(batches)>1:message+=f' (evidence batch {index+1}/{len(batches)})'
            git('-c','gc.auto=0','commit','-q','-m',message,env=env)
            git('reset','-q','HEAD','--',*OWNED);preserve()
            head=git('rev-parse','HEAD');remote=push_verified(head) if a.push else None
            append(ROOT/'state/publication_batches.jsonl',dict(time=time.time(),batch=index+1,batches=len(batches),
                commit=head,files=len(batch),raw_changed_bytes=sum((REPO/name).stat().st_size for name in batch),remote_SHA=remote,
                verified=remote==head if a.push else False,
                scope='Intermediate evidence commits are not a final release. All parts and reports are complete only at the final verified SHA.'))
    # Reconcile only this task's paths in the ordinary index.
    git('reset','-q','HEAD','--',*OWNED);preserve();head=git('rev-parse','HEAD')
    result=dict(time=time.time(),branch=BRANCH,commit=head,phase='FINAL' if a.final else 'CHECKPOINT',push='NOT_REQUESTED')
    if a.push:
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        if remote!=head:remote=push_verified(head)
        assert remote==head;result.update(push='VERIFIED',remote_SHA=remote)
    write(ROOT/'state'/('publish_'+('final' if a.final else 'checkpoint')+'.json'),result);print(json.dumps(result))


if __name__=='__main__':main()
