"""Publish only this campaign with a private Git index."""
import argparse,tempfile
from .common import *


DOCUMENTS=['dense_ar_dynamic_layer_pd_crossover_spec.md',
    'dense_ar_dynamic_layer_pd_implementation_and_failure_audit.md',
    'dense_ar_dynamic_layer_pd_crossover_final_summary.md',
    'dense_ar_pd_colocated_qps_crossover_analysis.md',
    'dense_ar_dynamic_layer_cost_model_validation.md',
    'dense_ar_dynamic_layer_decode_safe_serving_results.md',
    'dense_ar_slo_calibration_and_goodput_analysis.md']


def candidates():
    manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    archived={r['path'] for r in manifest['original_files']};selected=[];excluded=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        name=str(p.relative_to(REPO));reason=None
        if 'worker_source' in p.parts:reason='EXTRACTED_DUPLICATE_OF_VERIFIED_SOURCE_ZIP'
        elif '__pycache__' in p.parts:reason='REGENERABLE_PYTHON_CACHE'
        elif 'layer_compile_cache' in p.parts or any('suffix_cache' in part for part in p.parts):reason='REGENERABLE_COMPILER_CACHE'
        elif name in archived:reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif p.suffix in ('.lock','.local') or '.tmp.' in p.name:reason='LOCAL_OPERATIONAL_FILE'
        elif p.name in ('publication_receipt.json','publication_progress.json'):reason='POST_COMMIT_LOCAL_RECEIPT'
        elif p.name=='burner.log':reason='LIVE_LOG_HAS_CLOSEOUT_SNAPSHOT'
        elif p.suffix=='.bin' and 'state_transport' in p.parts:reason='EXPIRED_REGISTRATION_METADATA_HASHED_IN_DESCRIPTOR'
        if reason:excluded.append(dict(path=name,bytes=p.stat().st_size,reason=reason));continue
        assert not p.is_symlink() and p.stat().st_size<100*1024**2,name
        selected.append(p)
    selected+=sorted(p for p in SOURCE.rglob('*.py') if '__pycache__' not in p.parts)
    selected+=[REPO/'poc_flashvep/reports'/name for name in DOCUMENTS]
    assert all(p.exists() for p in selected)
    return sorted(set(selected)),excluded


def git_command(args,env=None,input=None):
    return subprocess.check_output(['git',*args],cwd=REPO,env=env,input=input,text=True).strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--push',action='store_true');a=parser.parse_args()
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists();preserve();before=preservation()
    paths,excluded=candidates();coverage=ROOT/'state/publication_coverage.json'
    write(coverage,dict(selected_files=len(paths),selected_bytes=sum(p.stat().st_size for p in paths),excluded=excluded,
        scope='The archive manifest identifies every lossless raw file. Per-job source ZIP files preserve extracted source. Original evidence remains local.'))
    paths.append(coverage);parent=git('rev-parse',BRANCH);commits=[];batches=[];batch=[];size=0
    for p in paths:
        if batch and size+p.stat().st_size>300*1024**2:batches.append(batch);batch=[];size=0
        batch.append(p);size+=p.stat().st_size
    if batch:batches.append(batch)
    with tempfile.TemporaryDirectory(prefix='dynamic_layer_private_index_') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'));git_command(['read-tree',parent],env)
        for i,batch in enumerate(batches):
            payload=b''.join(str(p.relative_to(REPO)).encode()+b'\0' for p in batch)
            r=subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],cwd=REPO,env=env,input=payload,capture_output=True)
            assert r.returncode==0,r.stderr.decode(errors='replace')[-4000:]
            tree=git_command(['write-tree'],env)
            if tree==git_command(['rev-parse',parent+'^{tree}']):continue
            message=f'Dense AR dynamic layer PD: measured campaign batch {i+1}/{len(batches)}\n\nPreserve exact state qualification, held-out performance, metrology limits, and failed experiments.\n'
            commit=git_command(['commit-tree',tree,'-p',parent],env,message)
            git_command(['update-ref','refs/heads/'+BRANCH,commit,parent]);parent=commit;commits.append(commit)
            if a.push:
                subprocess.run(['git','-c','pack.threads=2','push','origin',commit+':refs/heads/'+BRANCH],cwd=REPO,check=True)
                assert git_command(['ls-remote','origin','refs/heads/'+BRANCH]).split()[0]==commit
            write(ROOT/'state/publication_progress.json',dict(time=time.time(),branch=BRANCH,commits=commits,last_SHA=parent,
                pushed=a.push,finished_batches=i+1,total_batches=len(batches)))
            assert preservation()==before
            print('PUBLISHED_BATCH',i+1,len(batches),commit,flush=True)
    remote=git_command(['ls-remote','origin','refs/heads/'+BRANCH]).split()[0] if a.push else None
    if a.push:assert remote==parent
    preserve();assert preservation()==before
    write(ROOT/'state/publication_receipt.json',dict(time=time.time(),branch=BRANCH,SHA=parent,Remote_SHA=remote,
        remote_matches=remote==parent,pushed=a.push,commits=commits,user_checkout_and_index_preserved=True,
        external_runtime_preserved=True,scope='The final post-publication receipt remains local. Remote identity is checked independently.'))
    print(json.dumps(dict(branch=BRANCH,SHA=parent,Remote_SHA=remote)))


if __name__=='__main__':main()
