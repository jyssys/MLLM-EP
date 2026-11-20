"""Publish this campaign without modifying the user's checkout or index."""
import argparse
import collections
import gzip
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from .common import BASE,BRANCH,REPO,ROOT,SOURCE,digest,preserve,preservation,resources,write
from ..dense_ar_layer_token_pd.publication import git,stage

DOCUMENTS=('dense_ar_bi0_shortanswer200_layer_token_dynamic_final_summary.md',
    'dense_ar_shortanswer200_bi0_dataset_contract.md','dense_ar_bi0_native_agg_pd_validation.md',
    'dense_ar_bi0_prefill_heavy_safe_window_discovery.md',
    'dense_ar_bi0_layer_token_compute_cost_breakdown.md',
    'dense_ar_bi0_layer_token_sweetspot_knob_analysis.md',
    'dense_ar_bi0_dynamic_window_split_scheduler.md',
    'dense_ar_bi0_axis_selector_oracle_analysis.md','dense_ar_bi0_online_serving_qps_results.md',
    'dense_ar_bi0_transport_host_tail_analysis.md','dense_ar_bi0_research_decision_log.md',
    'dense_ar_bi0_shortanswer200_main_figures.md')


def probe_exclusion_reason(path):
    """Keep non-certified debug tensors local. Preserve their hashes."""
    if not (path.name.startswith('numerical_') and path.suffix=='.pt'):
        return None
    cfg=path.parent/'configuration.json'
    if not cfg.exists():return None
    value=json.loads(cfg.read_text())
    if path.name.startswith('numerical_prefill_') and value.get('axis_kind')=='Layer':
        return 'PARTIAL_PRODUCER_DEBUG_TENSOR_LOCAL_ONLY'
    if not value.get('numerical_probe_sync'):
        return 'UNSYNCHRONIZED_DEBUG_TENSOR_LOCAL_ONLY'
    return None


def restricted_probe(path):
    return probe_exclusion_reason(path) is not None


def archive():
    from ..dense_ar_dynamic_layer_pd import archive_evidence as proven
    for key,value in dict(ROOT=ROOT,REPO=REPO,digest=digest,write=write,resources=resources).items():
        setattr(proven,key,value)
    original=proven.selected_files
    def selected():
        extra=[p for p in (ROOT/'data').glob('*') if p.is_file() and p.stat().st_size>=1024**2]
        return sorted({p for p in original()+extra if not restricted_probe(p)},key=lambda p:(p.name,str(p)))
    proven.selected_files=selected
    proven.main()


def prepare():
    manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    assert manifest['verified_lossless']
    archived={r['path'] for r in manifest['original_files']}
    excluded=[];selected=[]
    metadata={'publication_coverage.json','publication_SHA256_manifest.json','publication_exclusions.jsonl.gz'}
    receipts={'publication_receipt.json','publication_progress.json','post_publication_final_check.json'}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        if path.name in metadata:continue
        relative=str(path.relative_to(REPO));reason=None
        if restricted_probe(path):reason=probe_exclusion_reason(path)
        elif 'worker_source' in path.parts:reason='EXTRACTED_DUPLICATE_OF_SOURCE_ZIP'
        elif '__pycache__' in path.parts or 'tools' in path.relative_to(ROOT).parts:reason='REGENERABLE_CACHE'
        elif relative in archived:reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif path.suffix=='.pdf' and 'references' in path.relative_to(ROOT).parts:reason='EXTERNAL_PAPER_NOT_REDISTRIBUTED'
        elif path.suffix in ('.local','.lock') or '.tmp.' in path.name:reason='LOCAL_OPERATIONAL_FILE'
        elif path.name in receipts:reason='POST_COMMIT_LOCAL_RECEIPT'
        elif path.name=='burner.log':reason='LIVE_LOG_WITH_CLOSEOUT_SNAPSHOT'
        elif path.suffix=='.bin' and 'state_transport' in path.parts:reason='EXPIRED_TRANSPORT_DESCRIPTOR'
        if reason:
            row=dict(path=relative,bytes=path.stat().st_size,reason=reason)
            if restricted_probe(path) or reason=='EXPIRED_TRANSPORT_DESCRIPTOR':row['SHA256']=digest(path)
            excluded.append(row)
        else:selected.append(path)
    selected.extend(SOURCE.glob('*.py'))
    selected.extend(REPO/'poc_flashvep/reports'/name for name in DOCUMENTS)
    for suffix in ('png','pdf'):
        selected.extend((REPO/'poc_flashvep').glob(f'bi0_shortanswer200*.{suffix}'))
    assert all(p.exists() and not p.is_symlink() and p.stat().st_size<100*1024**2 for p in selected)
    ledger=ROOT/'state/publication_exclusions.jsonl.gz'
    with gzip.open(ledger,'wt',encoding='utf-8') as stream:
        for row in excluded:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    payload=sorted(set(selected+[ledger]));coverage=ROOT/'state/publication_coverage.json'
    write(coverage,dict(source_parent=BASE,selected_files=len(payload)+2,
        payload_bytes=sum(p.stat().st_size for p in payload),excluded_files=len(excluded),
        excluded_reason_counts=dict(collections.Counter(r['reason'] for r in excluded)),
        scope='Only this BI0 namespace, source, reports and figures. Original evidence remains local. '
        'Partial-producer and unsynchronized debug tensors remain local with exclusion hashes. '
        'Synchronized valid teacher and destination tensors remain included. Mismatch reports and failed logs remain included.'))
    hashes=ROOT/'state/publication_SHA256_manifest.json'
    write(hashes,dict(files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p)) for p in payload+[coverage]],
        self_included=False,scope='The Git tree binds this manifest. Decompressed archive members have separate SHA256 checks.'))
    return sorted(set(payload+[coverage,hashes]))


def proof():
    preserve();paths=[SOURCE/'publication.py',ROOT/'configs/frozen_fast_shared_runtime_v3.json']
    with tempfile.TemporaryDirectory(prefix='layer_token_private_bi0_proof_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'))
        git(['read-tree','--empty'],env)
        git(['add','-f','--pathspec-from-file=-','--pathspec-file-nul'],env,
            b''.join(str(p.relative_to(REPO)).encode()+b'\0' for p in paths))
        expected=git(['write-tree'],env).decode().strip()
        git(['read-tree','--empty'],env);stage(paths,env)
        observed=git(['write-tree'],env).decode().strip()
    assert expected==observed;preserve()
    write(ROOT/'state/private_staging_tree_proof.json',dict(passed=True,expected_tree=expected,observed_tree=observed))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true')
    parser.add_argument('--selftest',action='store_true');parser.add_argument('--push',action='store_true')
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.selftest:proof();return
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists();preserve()
    if args.archive:archive();return
    before=preservation();paths=prepare()
    if args.prepare:print('PUBLICATION_PREPARED',len(paths),flush=True);return
    parent=git(['rev-parse',BRANCH]).decode().strip()
    assert parent==BASE or git(['merge-base',BASE,parent]).decode().strip()==BASE
    chunks=[];current=[];size=0
    for path in sorted(paths,key=lambda p:('large_evidence.tar.zst.part' in p.name,str(p))):
        if current and size+path.stat().st_size>256*1024**2:chunks.append(current);current=[];size=0
        current.append(path);size+=path.stat().st_size
    if current:chunks.append(current)
    progress=[]
    with tempfile.TemporaryDirectory(prefix='layer_token_private_bi0_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'));git(['read-tree',parent],env)
        for index,chunk in enumerate(chunks):
            stage(chunk,env);tree=git(['write-tree'],env).decode().strip()
            if tree!=git(['rev-parse',parent+'^{tree}']).decode().strip():
                message=f'BI0 ShortAnswer200 PD evidence {index+1}/{len(chunks)}\n\nKeep numerical geometry, measured costs and safety limits separate.\n'.encode()
                commit=git(['commit-tree',tree,'-p',parent],env,message).decode().strip()
                git(['update-ref','refs/heads/'+BRANCH,commit,parent]);parent=commit
            if args.push:
                result=subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false','-c','pack.threads=2',
                    'push','origin',parent+':refs/heads/'+BRANCH],cwd=REPO,capture_output=True,text=True)
                progress.append(dict(time=time.time(),checkpoint=index+1,SHA=parent,returncode=result.returncode,
                    stdout=result.stdout,stderr=result.stderr))
                write(ROOT/'state/publication_progress.json',progress);assert result.returncode==0,result.stderr[-4000:]
            print('PUBLICATION_CHECKPOINT',index+1,len(chunks),parent,flush=True)
    remote=git(['ls-remote','origin','refs/heads/'+BRANCH]).decode().split()
    remote_sha=remote[0] if remote else None
    if args.push:assert remote_sha==parent
    preserve();assert preservation()==before
    write(ROOT/'state/publication_receipt.json',dict(time=time.time(),branch=BRANCH,SHA=parent,Remote_SHA=remote_sha,
        pushed=args.push,remote_matches=remote_sha==parent,user_checkout_and_index_preserved=True,external_runtime_preserved=True))
    print(json.dumps(dict(branch=BRANCH,SHA=parent,Remote_SHA=remote_sha)),flush=True)


if __name__=='__main__':main()
