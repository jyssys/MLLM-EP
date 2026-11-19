"""Publish Native evidence with a private index and verified archive members."""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from .common import BASE,BRANCH,REPO,ROOT,SOURCE,digest,preserve,preservation,write,resources
from ..dense_ar_layer_token_pd.publication import git,stage


DOCUMENTS=('dense_ar_longbench200_native_pd_aggdp2_final_summary.md',
    'dense_ar_previous_experiment_safe_stop.md','dense_ar_longbench_v1_200_workload_contract.md',
    'dense_ar_longbench200_dataset_contract.md','dense_ar_native_pd_reproduction_validation.md',
    'dense_ar_native_decode_latency_root_cause.md','dense_ar_official_vllm_reference_comparison.md',
    'dense_ar_longbench200_qps_sweep_results.md','dense_ar_aggdp2_native_pd_crossover_analysis.md',
    'dense_ar_longbench200_prefill_heavy_feasibility.md')


def archive():
    from ..dense_ar_dynamic_layer_pd import archive_evidence as proven
    for key,value in dict(ROOT=ROOT,REPO=REPO,digest=digest,write=write,resources=resources).items():
        setattr(proven,key,value)
    original=proven.selected_files
    def selected():
        data=[p for p in (ROOT/'data').glob('*') if p.is_file() and p.stat().st_size>=1024**2]
        return sorted(set(original())|set(data),key=lambda p:(p.name,str(p)))
    proven.selected_files=selected;proven.main()


def prepare():
    archive_manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    assert archive_manifest['verified_lossless']
    archived={r['path'] for r in archive_manifest['original_files']}
    excluded=[];selected=[]
    metadata={'publication_coverage.json','publication_SHA256_manifest.json','publication_exclusions.jsonl.gz'}
    receipts={'publication_receipt.json','publication_progress.json','post_publication_final_check.json'}
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        relative=str(p.relative_to(REPO));reason=None
        if p.name in metadata:continue
        if 'worker_source' in p.parts:reason='EXTRACTED_DUPLICATE_OF_SOURCE_ZIP'
        elif '__pycache__' in p.parts or 'tools' in p.relative_to(ROOT).parts:reason='REGENERABLE_CACHE'
        elif relative in archived:reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif p.suffix=='.pdf' and 'references' in p.relative_to(ROOT).parts:reason='EXTERNAL_PAPER_NOT_REDISTRIBUTED'
        elif p.suffix in ('.local','.lock') or '.tmp.' in p.name:reason='LOCAL_OPERATIONAL_FILE'
        elif p.name in receipts:reason='POST_COMMIT_LOCAL_RECEIPT'
        elif p.name=='burner.log':reason='LIVE_LOG_WITH_CLOSEOUT_SNAPSHOT'
        if reason:excluded.append(dict(path=relative,bytes=p.stat().st_size,reason=reason))
        else:selected.append(p)
    selected.extend(SOURCE.glob('*.py'))
    selected.extend(REPO/'poc_flashvep/reports'/name for name in DOCUMENTS)
    for suffix in ('png','pdf'):selected.extend((REPO/'poc_flashvep').glob(f'longbench200_*_vs_qps.{suffix}'))
    assert all(p.exists() and not p.is_symlink() and p.stat().st_size<100*1024**2 for p in selected)
    ledger=ROOT/'state/publication_exclusions.jsonl.gz'
    with gzip.open(ledger,'wt',encoding='utf-8') as stream:
        for row in excluded:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    payload=sorted(set(selected+[ledger]));coverage=ROOT/'state/publication_coverage.json'
    write(coverage,dict(source_parent=BASE,selected_files=len(payload)+2,
        payload_bytes=sum(p.stat().st_size for p in payload),excluded_files=len(excluded),
        excluded_reason_counts=dict(collections.Counter(r['reason'] for r in excluded)),
        scope='Only the new Native namespace, reports and figures. Archive originals remain local. User checkout, index and external runtime remain unchanged.'))
    manifest=ROOT/'state/publication_SHA256_manifest.json'
    write(manifest,dict(files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p)) for p in payload+[coverage]],
        self_included=False,scope='The Git tree binds this manifest. Archive member hashes are verified separately. Live logs and post-commit receipts are excluded explicitly.'))
    return sorted(set(payload+[coverage,manifest]))


def proof():
    preserve();paths=[SOURCE/'publication.py',ROOT/'configs/frozen_LongBench200_Main_grid_v1.json']
    with tempfile.TemporaryDirectory(prefix='layer_token_private_native_proof_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'))
        git(['read-tree','--empty'],env)
        git(['add','-f','--pathspec-from-file=-','--pathspec-file-nul'],env,
            b''.join(str(p.relative_to(REPO)).encode()+b'\0' for p in paths))
        expected=git(['write-tree'],env).decode().strip()
        git(['read-tree','--empty'],env);stage(paths,env)
        observed=git(['write-tree'],env).decode().strip()
    assert observed==expected;preserve()
    write(ROOT/'state/private_staging_tree_proof.json',dict(passed=True,expected_tree=expected,observed_tree=observed))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true')
    parser.add_argument('--selftest',action='store_true');parser.add_argument('--push',action='store_true')
    args=parser.parse_args()
    if args.selftest:proof();return
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists();preserve()
    if args.archive:archive();return
    before=preservation();paths=prepare();parent=git(['rev-parse',BRANCH]).decode().strip()
    assert parent==BASE or git(['merge-base',BASE,parent]).decode().strip()==BASE
    chunks=[];current=[];size=0
    for p in sorted(paths,key=lambda p:('large_evidence.tar.zst.part' in p.name,str(p))):
        if current and size+p.stat().st_size>256*1024**2:chunks.append(current);current=[];size=0
        current.append(p);size+=p.stat().st_size
    if current:chunks.append(current)
    progress=[]
    with tempfile.TemporaryDirectory(prefix='layer_token_private_native_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'));git(['read-tree',parent],env)
        for index,chunk in enumerate(chunks):
            stage(chunk,env);tree=git(['write-tree'],env).decode().strip()
            if tree!=git(['rev-parse',parent+'^{tree}']).decode().strip():
                message=f'Dense AR Native PD and LongBench200 evidence {index+1}/{len(chunks)}\n\nPreserve numerical-path controls, qualified outputs and finite-cohort limits.\n'.encode()
                commit=git(['commit-tree',tree,'-p',parent],env,message).decode().strip()
                git(['update-ref','refs/heads/'+BRANCH,commit,parent]);parent=commit
            if args.push:
                r=subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false','-c','pack.threads=2','push',
                    'origin',parent+':refs/heads/'+BRANCH],cwd=REPO,capture_output=True,text=True)
                progress.append(dict(time=time.time(),checkpoint=index+1,SHA=parent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
                write(ROOT/'state/publication_progress.json',progress);assert r.returncode==0,r.stderr[-4000:]
            print('PUBLICATION_CHECKPOINT',index+1,len(chunks),parent,flush=True)
    remote=git(['ls-remote','origin','refs/heads/'+BRANCH]).decode().split()
    remote_sha=remote[0] if remote else None
    if args.push:assert remote_sha==parent
    preserve();assert preservation()==before
    write(ROOT/'state/publication_receipt.json',dict(time=time.time(),branch=BRANCH,SHA=parent,Remote_SHA=remote_sha,
        pushed=args.push,remote_matches=remote_sha==parent,user_checkout_and_index_preserved=True,external_runtime_preserved=True))
    print(json.dumps(dict(branch=BRANCH,SHA=parent,Remote_SHA=remote_sha)),flush=True)


if __name__=='__main__':main()
