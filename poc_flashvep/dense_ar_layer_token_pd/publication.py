"""Publish only the new namespace. Preserve the user's Git index."""
import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from .common import BASE,BRANCH,REPO,ROOT,SOURCE,digest,preserve,preservation,write

DOCUMENTS=('dense_ar_layer_token_dynamic_headroom_spec.md',
    'dense_ar_layer_token_dynamic_headroom_final_summary.md',
    'dense_ar_tp2_numerical_correctness_root_cause.md',
    'dense_ar_layer_vs_token_correctness_and_execution.md',
    'dense_ar_layer_vs_token_online_results.md',
    'dense_ar_dynamic_vs_static_oracle_and_online_results.md',
    'dense_ar_safe_budget_variance_analysis.md',
    'dense_ar_layer_token_dynamic_main_figures.md')


def git(args,env=None,payload=None):
    result=subprocess.run(['git',*args],cwd=REPO,env=env,input=payload,capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors='replace')[-4000:]
    return result.stdout


def archive():
    from ..dense_ar_dynamic_layer_pd import archive_evidence as previous
    from .common import resources
    for key,value in dict(ROOT=ROOT,REPO=REPO,digest=digest,write=write,resources=resources).items():
        setattr(previous,key,value)
    original=previous.selected_files
    def selected():
        extra=[p for p in (ROOT/'data').glob('*') if p.is_file() and p.stat().st_size>=1024**2]
        return sorted(set(original())|set(extra),key=lambda p:(p.name,str(p)))
    previous.selected_files=selected
    previous.main()


def candidates():
    manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    archived={row['path'] for row in manifest['original_files']}
    selected=[];excluded=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative=str(path.relative_to(REPO));reason=None
        if 'worker_source' in path.parts:
            reason='EXTRACTED_DUPLICATE_OF_VERIFIED_SOURCE_ZIP'
        elif '__pycache__' in path.parts or 'tools' in path.relative_to(ROOT).parts:
            reason='REGENERABLE_CACHE_OR_TOOL_NOT_PUBLISHED'
        elif relative in archived:
            reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif path.suffix=='.pdf' and 'references' in path.relative_to(ROOT).parts:
            reason='EXTERNAL_PAPER_NOT_REDISTRIBUTED'
        elif path.suffix in ('.lock','.local') or '.tmp.' in path.name:
            reason='LOCAL_OPERATIONAL_FILE'
        elif path.name in ('publication_receipt.json','publication_progress.json','push_command_result.json',
                           'post_publication_final_check.json'):
            reason='POST_COMMIT_LOCAL_RECEIPT'
        elif path.name=='burner.log':
            reason='LIVE_LOG_CLOSEOUT_SNAPSHOT_PRESERVED'
        elif path.suffix=='.bin' and 'state_transport' in path.parts:
            reason='EXPIRED_TRANSPORT_REGISTRATION_DESCRIPTOR_HASH_PRESERVED'
        if reason:
            record=dict(path=relative,bytes=path.stat().st_size,reason=reason)
            if reason=='EXPIRED_TRANSPORT_REGISTRATION_DESCRIPTOR_HASH_PRESERVED':
                record['SHA256']=digest(path)
            excluded.append(record)
        else:
            assert not path.is_symlink() and path.stat().st_size<100*1024**2,relative
            selected.append(path)
    selected.extend(SOURCE.glob('*.py'))
    selected.extend(REPO/'poc_flashvep/reports'/name for name in DOCUMENTS)
    selected.extend(REPO/'poc_flashvep'/('dense_ar_layer_token_dynamic_'+stem+suffix)
        for stem in ('mean_ttft','mean_tbt','output_tokens_s','slo') for suffix in ('.png','.pdf'))
    assert all(path.exists() for path in selected)
    return sorted(set(selected)),excluded


def stage(paths,env):
    assert 'layer_token_private_' in env['GIT_INDEX_FILE']
    names=[str(path.relative_to(REPO)) for path in paths]
    assert all(name.isascii() and not any(c.isspace() for c in name) for name in names)
    hashes=git(['hash-object','-w','--stdin-paths'],env,''.join(name+'\n' for name in names).encode()).decode().splitlines()
    assert len(names)==len(hashes)
    entries=[]
    for path,name,sha in zip(paths,names,hashes):
        assert not path.is_symlink() and stat.S_ISREG(path.stat().st_mode)
        mode='100755' if path.stat().st_mode&0o111 else '100644'
        entries.append(f'{mode} {sha}\t{name}\0'.encode())
    git(['update-index','--add','-z','--index-info'],env,b''.join(entries))


def selftest():
    preserve()
    paths=[SOURCE/'publication.py',ROOT/'configs/frozen_SLO_regimes.json']
    with tempfile.TemporaryDirectory(prefix='layer_token_private_proof_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'))
        git(['read-tree','--empty'],env)
        git(['add','-f','--pathspec-from-file=-','--pathspec-file-nul'],env,
            b''.join(str(path.relative_to(REPO)).encode()+b'\0' for path in paths))
        expected=git(['write-tree'],env).decode().strip()
        git(['read-tree','--empty'],env);stage(paths,env)
        observed=git(['write-tree'],env).decode().strip()
    assert expected==observed
    preserve()
    write(ROOT/'state/private_staging_tree_proof.json',dict(time=time.time(),passed=True,
        expected_tree=expected,observed_tree=observed,user_index_untouched=True))
    print('PRIVATE_INDEX_PROOF',observed,flush=True)


def prepare():
    paths,excluded=candidates()
    coverage=ROOT/'state/publication_coverage.json'
    manifest=ROOT/'state/publication_SHA256_manifest.json'
    payload=[path for path in paths if path not in (coverage,manifest)]
    write(coverage,dict(source_parent=BASE,selected_files=len(payload)+2,
        payload_bytes_excluding_publication_metadata=sum(p.stat().st_size for p in payload),
        excluded=excluded,
        scope='Publish new source, new reports and evidence only. Every archived member has a verified decompressed SHA256. Originals remain local. No checkout, reset, force push or user-index update occurs. The count includes this coverage file and the SHA256 manifest. Payload bytes exclude these two metadata files.'))
    hashed=sorted(payload+[coverage])
    write(manifest,dict(files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p))
        for p in hashed],self_included=False,
        archived_originals_manifest=str((ROOT/'archives/large_evidence_manifest.json').relative_to(REPO)),
        scope='SHA256 covers each published payload file and publication coverage. The manifest excludes itself. The final Git commit binds the manifest. Archive parts and their verified original-member manifest are both covered. Post-commit local receipts and the live burner log are excluded explicitly.'))
    paths=sorted(set(payload+[coverage,manifest]))
    assert all(p.stat().st_size<100*1024**2 for p in paths),'A metadata file exceeds the server file limit'
    return paths


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',action='store_true')
    parser.add_argument('--push',action='store_true')
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.selftest:
        selftest();return
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    preserve()
    if args.archive:
        archive();return
    before=preservation();paths=prepare()
    if args.prepare:
        preserve();print('PUBLICATION_SHA256_PREPARED',len(paths),flush=True);return
    parent=git(['rev-parse',BRANCH]).decode().strip()
    assert parent==BASE or git(['merge-base',BASE,parent]).decode().strip()==BASE
    # Incremental fast-forward pushes keep each archive upload below the server
    # pack-size limit. Partial commits are explicit evidence-upload checkpoints.
    chunks=[];current=[];size=0
    ordered=sorted(paths,key=lambda p:('large_evidence.tar.zst.part' in p.name,str(p)))
    for path in ordered:
        if current and size+path.stat().st_size>256*1024**2:
            chunks.append(current);current=[];size=0
        current.append(path);size+=path.stat().st_size
    if current:
        chunks.append(current)
    progress_path=ROOT/'state/publication_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else []
    with tempfile.TemporaryDirectory(prefix='layer_token_private_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'))
        git(['read-tree',parent],env)
        for index,chunk in enumerate(chunks):
            stage(chunk,env);tree=git(['write-tree'],env).decode().strip()
            if tree!=git(['rev-parse',parent+'^{tree}']).decode().strip():
                message=f'Dense AR layer/token evidence checkpoint {index+1}/{len(chunks)}\n\nPreserve strict parity failures, independent starts and matched controls.\n'.encode()
                commit=git(['commit-tree',tree,'-p',parent],env,message).decode().strip()
                git(['update-ref','refs/heads/'+BRANCH,commit,parent]);parent=commit
            if args.push:
                result=subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false','-c','pack.threads=2','push','origin',
                    parent+':refs/heads/'+BRANCH],cwd=REPO,capture_output=True,text=True)
                progress.append(dict(time=time.time(),checkpoint=index+1,SHA=parent,
                    returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
                write(ROOT/'state/publication_progress.json',progress)
                assert result.returncode==0,result.stderr[-4000:]
            print('PUBLICATION_CHECKPOINT',index+1,len(chunks),parent,flush=True)
    remote=git(['ls-remote','origin','refs/heads/'+BRANCH]).decode().split()
    remote_sha=remote[0] if remote else None
    if args.push:
        assert remote_sha==parent
    preserve();assert before==preservation()
    write(ROOT/'state/publication_receipt.json',dict(time=time.time(),branch=BRANCH,
        SHA=parent,Remote_SHA=remote_sha,pushed=args.push,remote_matches=remote_sha==parent,
        user_checkout_and_index_preserved=True,external_runtime_preserved=True,
        scope='Private-index publication. This post-commit receipt remains local.'))
    print(json.dumps(dict(branch=BRANCH,SHA=parent,Remote_SHA=remote_sha)),flush=True)


if __name__=='__main__':
    main()
