"""Publish only campaign files without changing the user's checkout or index."""
import argparse,json,os,subprocess,tempfile,time
from .common import *


DOCUMENTS=[
    'poc_flashvep/reports/dense_ar_prefill_heavy_layerwise_pd_feasibility_spec.md',
    'poc_flashvep/reports/dense_ar_layerwise_pd_prior_work_runtime_review.md',
    'poc_flashvep/reports/dense_ar_prefill_heavy_layerwise_pd_feasibility_final_summary.md',
]


def candidates():
    manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    archived={r['path'] for r in manifest['original_files']}
    selected=[];excluded=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        name=str(p.relative_to(REPO));reason=None
        if 'worker_source' in p.parts:reason='EXTRACTED_SOURCE_DUPLICATES_SOURCE_ZIP'
        elif '__pycache__' in p.parts:reason='REGENERABLE_PYTHON_CACHE'
        elif 'layer_compile_cache' in p.parts or 'separate_suffix_cache_v14' in p.parts:reason='REGENERABLE_COMPILER_CACHE'
        elif name in archived:reason='VERIFIED_LOSSLESS_ARCHIVE_PARTS'
        elif p.suffix in ('.lock','.local') or '.tmp.' in p.name:reason='LOCAL_OPERATIONAL_FILE'
        elif p.name in ('private_runtime.patch','publication_progress.json','publication_receipt.json'):
            reason='PRIVATE_RUNTIME_DIFF_OR_POST_COMMIT_RECEIPT'
        elif p.name=='burner.log':reason='LIVE_LOG_HAS_CLOSEOUT_SNAPSHOT'
        elif p.suffix=='.bin' and 'state_transport' in p.parts:reason='EXPIRED_NIXL_AGENT_METADATA_HASH_RETAINED_IN_DESCRIPTORS'
        if reason:
            # Regenerable directories already have frozen source and runtime hashes.
            excluded.append(dict(path=name,bytes=p.stat().st_size,reason=reason));continue
        assert not p.is_symlink() and p.stat().st_size<100*1024**2,name
        selected.append(p)
    selected +=sorted(p for p in SOURCE.rglob('*.py') if '__pycache__' not in p.parts)
    selected +=[REPO/p for p in DOCUMENTS]
    assert all(p.exists() for p in selected)
    return sorted(set(selected)),excluded


def command(args,env=None,input=None):
    return subprocess.check_output(['git',*args],cwd=REPO,env=env,input=input,text=True).strip()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--push',action='store_true');args=parser.parse_args()
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    preserve();before=preservation();paths,excluded=candidates()
    coverage=ROOT/'state/publication_coverage.json'
    write(coverage,dict(selected_files=len(paths),selected_bytes=sum(p.stat().st_size for p in paths),
        excluded=excluded,archive_manifest='archives/large_evidence_manifest.json',
        scope='All original evidence remains local. Large raw tensors use verified lossless archive parts. Per-job source ZIP files preserve extracted worker_source. Compiler caches can be regenerated. Expired transport registration blobs are not reusable model state.'))
    paths.append(coverage)
    def priority(path):
        relative=path.relative_to(REPO)
        important=path in [REPO/p for p in DOCUMENTS] or SOURCE in path.parents or any(
            ROOT/k in path.parents for k in ('analysis','configs','state','figures'))
        return (0 if important else 1,str(relative))
    paths=sorted(set(paths),key=priority)
    parent=git('rev-parse',BRANCH);receipt=ROOT/'state/publication_progress.json'
    commits=[];batch=[];bytes_in_batch=0;batches=[]
    for path in paths:
        size=path.stat().st_size
        if batch and bytes_in_batch+size>350*1024**2:
            batches.append(batch);batch=[];bytes_in_batch=0
        batch.append(path);bytes_in_batch+=size
    if batch:batches.append(batch)
    with tempfile.TemporaryDirectory(prefix='dense_ar_private_index_') as temp:
        index=Path(temp)/'index';env=dict(os.environ,GIT_INDEX_FILE=str(index))
        command(['read-tree',parent],env)
        for i,batch in enumerate(batches):
            relative=[str(p.relative_to(REPO)) for p in batch]
            result=subprocess.run(['git','add','-f','--pathspec-from-file=-','--pathspec-file-nul'],
                cwd=REPO,env=env,input=b''.join(p.encode()+b'\0' for p in relative),capture_output=True)
            assert result.returncode==0,result.stderr.decode(errors='replace')[-4000:]
            tree=command(['write-tree'],env)
            if tree==command(['rev-parse',parent+'^{tree}']):continue
            message=f'Dense AR layer-wise PD feasibility: preserved evidence batch {i+1}/{len(batches)}\n\nKeep Native correctness, exact state qualification, and clean performance gates separate. Preserve the user index and external runtime.\n'
            commit=command(['commit-tree',tree,'-p',parent],env,message)
            command(['update-ref','refs/heads/'+BRANCH,commit,parent])
            parent=commit;commits.append(commit)
            if args.push:
                subprocess.run(['git','-c','pack.threads=2','push','origin',commit+':refs/heads/'+BRANCH],cwd=REPO,check=True)
                remote=command(['ls-remote','origin','refs/heads/'+BRANCH]).split()[0]
                assert remote==commit
            write(receipt,dict(time=time.time(),branch=BRANCH,commits=commits,last_SHA=parent,
                pushed=args.push,finished_batches=i+1,total_batches=len(batches),preservation=preservation()))
            assert preservation()==before
            print('PUBLICATION_BATCH',i+1,len(batches),commit,args.push,flush=True)
    assert preservation()==before;preserve()
    remote=command(['ls-remote','origin','refs/heads/'+BRANCH]).split()[0] if args.push else None
    if args.push:assert remote==parent
    write(ROOT/'state/publication_receipt.json',dict(time=time.time(),branch=BRANCH,SHA=parent,
        Remote_SHA=remote,remote_matches=remote==parent,pushed=args.push,commits=commits,
        user_checkout_and_index_preserved=True,external_runtime_preserved=True,
        scope='This post-publication receipt remains local. Remote branch identity is independently verified.'))
    print(json.dumps(dict(branch=BRANCH,SHA=parent,Remote_SHA=remote)))


if __name__=='__main__':main()
