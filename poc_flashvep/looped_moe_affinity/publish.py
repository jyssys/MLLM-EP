"""Commit only this study. Preserve the user's unrelated index exactly."""
import argparse
import hashlib
import json
import os
import subprocess
from .common import REPO,ROOT,write

SCOPE=['poc_flashvep/looped_moe_affinity','poc_flashvep/reports/looped_moe_ep_affinity_placement',
    'poc_flashvep/reports/looped_moe_ep_affinity_placement_poc_spec.md']
BRANCH='flashvep/looped-moe-ep-affinity-placement-poc'
BASE='a5b37f72245969782a77c93a1b0f0bb700c3494f'

def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True,
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0'),timeout=1800)

def unrelated_index_hash():
    payload=subprocess.check_output(['git','diff','--cached','--binary','--','.',*[':(exclude)'+path for path in SCOPE]],cwd=REPO)
    return hashlib.sha256(payload).hexdigest()

def in_scope(path):
    return any(path==prefix or path.startswith(prefix+'/') for prefix in SCOPE)

def reviewed_head():
    """Accept the original base or the exact checkpoint that this publisher verified."""
    head=git('rev-parse','HEAD').strip()
    if head==BASE:return head
    path=ROOT/'checkpoint_release_state.json'
    assert path.is_file(),'An intervening commit requires an explicit release review.'
    state=json.loads(path.read_text())
    assert state.get('status')=='COMMITTED_AND_PUSHED','The checkpoint push is not verified.'
    assert state.get('base')==BASE and state.get('branch')==BRANCH,'The checkpoint has another release base.'
    assert state.get('commit')==head and state.get('remote_sha')==head,'An unreviewed commit follows the checkpoint.'
    git('merge-base','--is-ancestor',BASE,head)
    assert not git('rev-list','--merges',f'{BASE}..{head}').strip(),'Checkpoint history contains a merge.'
    for commit in git('rev-list',f'{BASE}..{head}').splitlines():
        paths=git('diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
        assert paths and all(in_scope(value) for value in paths),'Checkpoint history includes unrelated changes.'
    return head

def main(checkpoint=False):
    assert git('branch','--show-current').strip()==BRANCH,'Do not publish from an unrelated branch.'
    expected_head=reviewed_head()
    if checkpoint:
        assert 'IN_PROGRESS' in (ROOT/'checkpoint.md').read_text(),'A checkpoint must disclose pending work.'
    before=unrelated_index_hash()
    git('add','--',*SCOPE)
    assert before==unrelated_index_hash(),'The unrelated index changed.'
    assert git('branch','--show-current').strip()==BRANCH and git('rev-parse','HEAD').strip()==expected_head,'Repository changed during release preparation.'
    tracked=git('ls-files','--',*SCOPE).splitlines()
    for relative in tracked:
        path=REPO/relative
        assert path.is_file() and path.stat().st_size<95_000_000,relative
    message=('Checkpoint completed looped MoE EP evidence; experiments remain in progress' if checkpoint else
        'Study looped MoE EP affinity, quality reproduction, and E2E attribution')
    git('commit','--only','-m',message,
        '-m','Preserve complete native cells after the changing-logit allocator OOM. Validate expandable-segment recovery without changing model arithmetic.',
        '--',*SCOPE)
    assert before==unrelated_index_hash(),'The commit changed unrelated staged work.'
    commit=git('rev-parse','HEAD').strip()
    changed=git('diff-tree','--no-commit-id','--name-only','-r','HEAD').splitlines()
    assert all(in_scope(path) for path in changed)
    state_path=ROOT/('checkpoint_release_state.json' if checkpoint else 'release_state.json')
    report=ROOT/('checkpoint.md' if checkpoint else 'report.md')
    write(state_path,dict(status='COMMITTED_PUSH_PENDING',branch=BRANCH,commit=commit,base=BASE,
        kind='IN_PROGRESS_CHECKPOINT' if checkpoint else 'FINAL_RELEASE',
        unrelated_staged_diff_sha256=before,changed_files=len(changed),report=str(report)))
    git('push','origin',f'HEAD:refs/heads/{BRANCH}')
    remote=git('ls-remote','--heads','origin',f'refs/heads/{BRANCH}').split()[0]
    assert remote==commit,('REMOTE_SHA_MISMATCH',commit,remote)
    state=dict(status='COMMITTED_AND_PUSHED',branch=BRANCH,commit=commit,remote_sha=remote,base=BASE,
        kind='IN_PROGRESS_CHECKPOINT' if checkpoint else 'FINAL_RELEASE',
        unrelated_staged_diff_sha256=before,changed_files=len(changed),report=str(report))
    write(state_path,state)
    print(state,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',action='store_true',help='Publish completed evidence without declaring the study complete.')
    main(checkpoint=parser.parse_args().checkpoint)
