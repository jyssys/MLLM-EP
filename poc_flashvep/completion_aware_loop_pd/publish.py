"""Publish scoped artifacts with a private index and preserve the active checkout."""
import hashlib
import subprocess
import tempfile
from .common import *

REPORTS=REPO/'poc_flashvep/reports'
DOCUMENTS=[REPORTS/name for name in (
    'ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation_spec.md',
    'ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation_final_summary.md',
    'ouro_huginn_pd_idle_slack_loop_offloading_oracle_spec.md',
    'ouro_huginn_pd_idle_slack_loop_offloading_oracle_final_summary.md')]
INDEX_SHA='d0674d8fc531a451b003223020607e4b54dd264c5ca319a57ac387bd19a3e6b3'
PRIVATE=Path('/home/esjung/external/looped-pd-abc-rlt')
PRIVATE_SHA='4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7'


def git(*args,env=None,cwd=REPO):
    return subprocess.check_output(['git',*args],cwd=cwd,env=env).decode().strip()


def preserved():
    index=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)
    private=subprocess.check_output(['git','diff','--binary'],cwd=PRIVATE)
    assert hashlib.sha256(index).hexdigest()==INDEX_SHA,'The pre-existing index changed'
    assert hashlib.sha256(private).hexdigest()==PRIVATE_SHA,'The external runtime diff changed'


def main():
    assert all(p.exists() for p in DOCUMENTS)
    for path in DOCUMENTS:
        assert not any(token in path.read_text() for token in ('CONFIRMATION_PENDING','CLOSURE_PENDING'))
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    restored=json.loads((ROOT/'state/final_burner_verification.json').read_text())
    assert restored['all_eight_verified'] and restored['all_generation_started_and_completed_round']
    from .evidence_bundle import verify
    bundle=verify(ROOT/'evidence_bundle')
    preserved()
    active=git('branch','--show-current')
    previous=git('rev-parse','refs/heads/'+BRANCH)
    assert previous==BASE,'Do not replace an existing publication'
    names=[str(p.relative_to(REPO)) for p in SOURCE.rglob('*')
        if p.is_file() and p.suffix in ('.py','.cc') and '__pycache__' not in p.parts]
    names += [str(p.relative_to(REPO)) for p in DOCUMENTS]
    for path in ROOT.rglob('*'):
        if not path.is_file():continue
        relative=path.relative_to(ROOT)
        if any(part in ('raw','worker_source','__pycache__') for part in relative.parts):continue
        if path.suffix in ('.pyc','.lock') or path.name in ('burner.log','final_remote_verification.json'):continue
        names.append(str(path.relative_to(REPO)))
    names=sorted(set(names))
    assert all((REPO/name).stat().st_size<100_000_000 for name in names)
    with tempfile.TemporaryDirectory(prefix='completion_loop_publication_') as temporary:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temporary)/'index'))
        git('read-tree',BASE,env=env)
        for offset in range(0,len(names),100):
            git('add','-f','--',*names[offset:offset+100],env=env)
        changed=git('diff','--cached','--name-only',BASE,env=env).splitlines()
        assert changed and set(changed)<=set(names)
        code=[name for name in changed if name.endswith(('.py','.cc'))]
        if code:git('diff','--cached','--check',BASE,'--',*code,env=env)
        tree=git('write-tree',env=env)
        commit=git('commit-tree',tree,'-p',BASE,'-m',
            'Validate completion-aware adaptive Loop PD and Native idle Oracle limits',env=env)
        git('update-ref','refs/heads/'+BRANCH,commit,previous)
        git('push','origin',commit+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        assert remote==commit
        receipt=ROOT/'state/publication_receipt.json'
        write(receipt,dict(time=time.time(),branch=BRANCH,evidence_commit=commit,
            evidence_remote_SHA=remote,verified=True,files=len(names),
            index_entry_SHA256=INDEX_SHA,private_runtime_diff_SHA256=PRIVATE_SHA,
            bundle_original_bytes=bundle['original_bytes'],bundle_bytes=bundle['bundled_bytes'],
            scope='The metadata child records this verified evidence commit. The final SHA receives a separate remote check.'))
        git('add','-f','--',str(receipt.relative_to(REPO)),env=env)
        tree=git('write-tree',env=env)
        final=git('commit-tree',tree,'-p',commit,'-m','Record verified completion-aware PD publication',env=env)
        git('update-ref','refs/heads/'+BRANCH,final,commit)
        git('push','origin',final+':refs/heads/'+BRANCH)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        assert remote==final
        write(ROOT/'state/final_remote_verification.json',dict(time=time.time(),branch=BRANCH,
            commit=final,remote_SHA=remote,verified=True))
    preserved()
    assert git('branch','--show-current')==active
    print(json.dumps(dict(branch=BRANCH,commit=final,remote_SHA=remote,verified=True,files=len(names))))


if __name__=='__main__':
    main()
