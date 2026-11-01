"""Commit only this namespace with a temporary index. Never force-push."""
import argparse
import subprocess
import tempfile
from .common import *
from .bootstrap import preserve


def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env,text=True).strip()


def main():
    p=argparse.ArgumentParser();p.add_argument('--push',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args()
    preserve();assert git('branch','--show-current')==BRANCH
    files=list(SOURCE.rglob('*.py'))+[SPEC]+[REPO/'poc_flashvep/reports'/n for n in REPORTS if (REPO/'poc_flashvep/reports'/n).exists()]
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
    oversized=[f for f in files if (REPO/f).stat().st_size>=100_000_000]
    assert not oversized,('Publish lossless compressed evidence before GitHub push; preserve local originals.',oversized)
    with tempfile.TemporaryDirectory(prefix='atomic_pd_publish_') as directory:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(directory)/'index'));git('read-tree','HEAD',env=env)
        for i in range(0,len(files),100):git('add','-f','--',*files[i:i+100],env=env)
        staged=git('diff','--cached','--name-only',env=env).splitlines()
        assert all(any(f.startswith(p) if p.endswith('/') else f==p for p in OWNED) for f in staged)
        prose=[f for f in staged if (f.endswith('.py') or f.endswith('.md')) and '/raw/' not in f]
        if prose:git('diff','--cached','--check','--',*prose,env=env)
        if staged:git('commit','-q','-m','Record Atomic Prefill PD '+('final measured study' if a.final else 'implementation and qualification checkpoint'),env=env)
    # Reconcile only this task's paths in the ordinary index.
    git('reset','-q','HEAD','--',*OWNED);preserve();head=git('rev-parse','HEAD')
    result=dict(time=time.time(),branch=BRANCH,commit=head,phase='FINAL' if a.final else 'CHECKPOINT',push='NOT_REQUESTED')
    if a.push:
        git('push','-u','origin',BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        assert remote==head;result.update(push='VERIFIED',remote_SHA=remote)
    write(ROOT/'state/publish_'+('final' if a.final else 'checkpoint')+'.json',result);print(json.dumps(result))


if __name__=='__main__':main()
