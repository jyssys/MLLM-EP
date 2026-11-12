"""Publish this audit without changing the user's checkout or staged index."""
import argparse,tempfile
from .common import *


DOCUMENTS=[
    'poc_flashvep/reports/ouro_huginn_remote_ready_cross_depth_pd_poc_spec.md',
    'poc_flashvep/reports/ouro_huginn_remote_ready_cross_depth_pd_poc_final_summary.md',
]


def selected_files():
    paths=[]
    for folder in (SOURCE,ROOT):
        for path in sorted(folder.rglob('*')):
            if not path.is_file():continue
            if '__pycache__' in path.parts or '.tmp.' in path.name:continue
            if path.name in ('publication_receipt.json','publication_coverage.json','burner.log'):continue
            assert not path.is_symlink() and path.stat().st_size<100*1024**2,path
            paths.append(path)
    paths += [REPO/name for name in DOCUMENTS]
    assert all(path.exists() for path in paths)
    return sorted(set(paths))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--push',action='store_true');args=parser.parse_args()
    preserve();before=signatures()
    qualification=json.loads((ROOT/'state/verification.json').read_text())
    assert qualification['passed'] and qualification['all_eight_approved_burners_generating']
    paths=selected_files()
    coverage=ROOT/'state/publication_coverage.json'
    write(coverage,dict(files=[dict(path=str(path.relative_to(REPO)),bytes=path.stat().st_size,SHA256=digest(path)) for path in paths],
        original_input_archive='state/archived_input_verification.json',
        scope='Publish only the new audit, reports, and derived evidence. Preserve original input bytes in the verified parent.'))
    paths.append(coverage)
    parent=git('rev-parse',BRANCH)
    subprocess.run(['git','merge-base','--is-ancestor',PARENT,parent],cwd=REPO,check=True)
    with tempfile.TemporaryDirectory(prefix='remote_ready_private_index_') as temporary:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temporary)/'index'))
        def command(*arguments,input=None):
            return subprocess.check_output(['git',*arguments],cwd=REPO,env=env,input=input,text=True).strip()
        command('read-tree',parent)
        relative=[str(path.relative_to(REPO)) for path in paths]
        subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],cwd=REPO,env=env,
            input=b''.join(name.encode()+b'\0' for name in relative),check=True)
        tree=command('write-tree')
        if tree!=git('rev-parse',parent+'^{tree}'):
            message='Audit remote-ready cross-depth PD equivalence\n\nJoin historical transfer generations to actual Native Mixed batches. Stop at verified mechanism equivalence. Preserve unknown ready-work counts and all existing user state.\n'
            commit=command('commit-tree',tree,'-p',parent,input=message)
            command('update-ref','refs/heads/'+BRANCH,commit,parent)
        else:commit=parent
    assert signatures()==before
    remote=None
    if args.push:
        subprocess.run(['git','-c','pack.threads=2','push','origin',commit+':refs/heads/'+BRANCH],cwd=REPO,check=True,timeout=300)
        remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0]
        assert remote==commit
    preserve();assert signatures()==before
    write(ROOT/'state/publication_receipt.json',dict(time=time.time(),elapsed_wall_hours=(time.time()-START)/3600,
        branch=BRANCH,SHA=commit,Remote_SHA=remote,pushed=args.push,remote_matches=remote==commit,
        user_state_preserved=True,external_runtime_preserved=True,signatures=before,
        scope='This receipt remains local. The remote SHA was read after the push.'))
    print(json.dumps(dict(branch=BRANCH,SHA=commit,Remote_SHA=remote,pushed=args.push)))


if __name__=='__main__':main()
