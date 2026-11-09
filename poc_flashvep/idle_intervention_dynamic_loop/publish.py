"""Publish scoped evidence with a temporary index. Preserve the active checkout."""
import hashlib
import subprocess
import tempfile
from .common import *

FINAL=REPO/'poc_flashvep/reports/ouro_huginn_adaptive_idle_intervention_dynamic_loop_final_summary.md'
SPEC=FINAL.with_name('ouro_huginn_adaptive_idle_intervention_dynamic_loop_poc_spec.md')
INDEX_SHA='d0674d8fc531a451b003223020607e4b54dd264c5ca319a57ac387bd19a3e6b3'
PRIVATE=Path('/home/esjung/external/looped-pd-abc-rlt')
PRIVATE_SHA='4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7'


def git(*args,env=None,cwd=REPO):
    return subprocess.check_output(['git',*args],cwd=cwd,env=env).decode().strip()


def preserved():
    index=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)
    private=subprocess.check_output(['git','diff','--binary'],cwd=PRIVATE)
    assert hashlib.sha256(index).hexdigest()==INDEX_SHA
    assert hashlib.sha256(private).hexdigest()==PRIVATE_SHA


def main():
    assert FINAL.exists() and SPEC.exists()
    assert not any(marker in FINAL.read_text() for marker in ('CONFIRMATION_PENDING','CLOSURE_PENDING')),'Do not publish an unfinished report'
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    restored=json.loads((ROOT/'state/final_burner_verification.json').read_text())
    assert restored['all_eight_verified'] and restored['all_generation_started_and_completed_round']
    from .evidence_bundle import verify
    verify(ROOT/'evidence_bundle')
    preserved()
    active=git('branch','--show-current')
    previous=git('rev-parse','refs/heads/'+BRANCH)
    assert previous==BASE,'Do not replace another publication'
    names=[str(p.relative_to(REPO)) for p in SOURCE.rglob('*.py')]
    names += [str(FINAL.relative_to(REPO)),str(SPEC.relative_to(REPO))]
    for p in ROOT.rglob('*'):
        if not p.is_file():continue
        relative=p.relative_to(ROOT)
        if any(part in ('raw','worker_source','__pycache__') for part in relative.parts):continue
        if p.suffix in ('.pyc','.lock') or p.name in ('burner.log','final_remote_verification.json'):continue
        names.append(str(p.relative_to(REPO)))
    names=sorted(set(names))
    assert all((REPO/name).stat().st_size<100_000_000 for name in names)
    with tempfile.TemporaryDirectory(prefix='idle_loop_publication_') as temporary:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temporary)/'index'))
        git('read-tree',BASE,env=env)
        for offset in range(0,len(names),100):git('add','-f','--',*names[offset:offset+100],env=env)
        changed=git('diff','--cached','--name-only',BASE,env=env).splitlines()
        assert changed and set(changed)<=set(names)
        code=[name for name in changed if name.endswith('.py')]
        if code:git('diff','--cached','--check',BASE,'--',*code,env=env)
        tree=git('write-tree',env=env)
        commit=git('commit-tree',tree,'-p',BASE,'-m','Evaluate actual adaptive idle-time Loop and Token continuation',env=env)
        git('update-ref','refs/heads/'+BRANCH,commit,previous)
        git('push','origin',commit+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        assert remote==commit
        receipt=ROOT/'state/publication_receipt.json'
        write(receipt,dict(time=time.time(),branch=BRANCH,evidence_commit=commit,
            evidence_remote_SHA=remote,verified=True,files=len(names),
            index_entry_SHA256=INDEX_SHA,private_runtime_diff_SHA256=PRIVATE_SHA,
            scope='The metadata child records this verified evidence commit. The final SHA is verified separately.'))
        git('add','-f','--',str(receipt.relative_to(REPO)),env=env)
        tree=git('write-tree',env=env)
        final=git('commit-tree',tree,'-p',commit,'-m','Record verified adaptive continuation publication',env=env)
        git('update-ref','refs/heads/'+BRANCH,final,commit)
        git('push','origin',final+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        assert remote==final
        write(ROOT/'state/final_remote_verification.json',dict(time=time.time(),branch=BRANCH,
            commit=final,remote_SHA=remote,verified=True))
    preserved()
    assert git('branch','--show-current')==active
    print(json.dumps(dict(branch=BRANCH,commit=final,remote_SHA=remote,verified=True,files=len(names))))


if __name__=='__main__':main()
