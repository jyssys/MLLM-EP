"""Publish only this campaign. Preserve the user index and checkout."""
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
from ..dense_ar_bi0_shortanswer_pd.publication import probe_exclusion_reason

DOCUMENTS=('dense_ar_bi0_credit_queue_root_cause.md','dense_ar_bi0_all_cuts_qps_sweep.md',
    'dense_ar_bi0_input_length_axis_crossover.md','dense_ar_bi0_batch_kv_credit_sensitivity.md',
    'dense_ar_bi0_layer_token_sweetspot_knob_final.md','dense_ar_bi0_continuous_serving_bridge.md',
    'dense_ar_bi0_axis_root_cause_final_summary.md',
    'dense_ar_bi0_historical_layer_winner_reproduction.md',
    'dense_ar_bi0_ready_to_d_queue_root_cause.md',
    'dense_ar_bi0_attention_gemm_axis_crossover.md')


def archive():
    from ..dense_ar_dynamic_layer_pd import archive_evidence as proven
    for key,value in dict(ROOT=ROOT,REPO=REPO,digest=digest,write=write,resources=resources).items():setattr(proven,key,value)
    original=proven.selected_files
    def selected():
        extra=[p for p in (ROOT/'data').glob('*') if p.is_file() and p.stat().st_size>=1024**2]
        return sorted({p for p in original()+extra if probe_exclusion_reason(p) is None},key=str)
    proven.selected_files=selected;proven.main()


def prepare():
    manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text());assert manifest['verified_lossless']
    archived={r['path'] for r in manifest['original_files']};selected=[];excluded=[]
    local={'publication_receipt.json','publication_progress.json','post_publication_final_check.json'}
    generated={'publication_coverage.json','publication_SHA256_manifest.json','publication_exclusions.jsonl.gz'}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name in generated:continue
        relative=str(path.relative_to(REPO));reason=None
        if probe_exclusion_reason(path):reason=probe_exclusion_reason(path)
        elif 'worker_source' in path.parts:reason='DUPLICATE_OF_FROZEN_SOURCE_ZIP'
        elif '__pycache__' in path.parts or 'tools' in path.relative_to(ROOT).parts:reason='REGENERABLE_CACHE'
        elif relative in archived:reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif path.name=='burner.log':reason='LIVE_LOG_WITH_CLOSEOUT_SNAPSHOT'
        elif path.name in local:reason='POST_COMMIT_LOCAL_RECEIPT'
        elif path.suffix=='.bin' and ('hidden_transport' in path.parts or 'state_transport' in path.parts):reason='EXPIRED_TRANSPORT_METADATA'
        elif path.suffix in ('.local','.lock') or '.tmp.' in path.name:reason='LOCAL_OPERATIONAL_FILE'
        if reason:excluded.append(dict(path=relative,bytes=path.stat().st_size,SHA256=digest(path),reason=reason))
        else:selected.append(path)
    selected.extend(SOURCE.glob('*.py'));selected.extend(SOURCE.glob('*.cc'))
    selected.extend(REPO/'poc_flashvep/reports'/name for name in DOCUMENTS)
    assert all(p.exists() and not p.is_symlink() and p.stat().st_size<100*1024**2 for p in selected)
    ledger=ROOT/'state/publication_exclusions.jsonl.gz'
    with gzip.open(ledger,'wt',encoding='utf-8') as stream:
        for row in excluded:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    payload=sorted(set(selected+[ledger]));coverage=ROOT/'state/publication_coverage.json'
    write(coverage,dict(base=BASE,selected_files=len(payload)+2,payload_bytes=sum(p.stat().st_size for p in payload),
        exclusion_counts=dict(collections.Counter(r['reason'] for r in excluded)),
        scope='New source, reports, figures and evidence only. Failed runs stay included. Archive members have lossless verification and hashes. Existing user files are not staged.'))
    hashes=ROOT/'state/publication_SHA256_manifest.json'
    write(hashes,dict(files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p)) for p in payload+[coverage]],self_included=False))
    return sorted(set(payload+[coverage,hashes]))


def proof():
    preserve();paths=[SOURCE/'publication.py',ROOT/'configs/frozen_research_design_v1.json']
    with tempfile.TemporaryDirectory(prefix='layer_token_private_axis_rootcause_proof_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'))
        git(['read-tree','--empty'],env)
        git(['add','-f','--pathspec-from-file=-','--pathspec-file-nul'],env,b''.join(str(p.relative_to(REPO)).encode()+b'\0' for p in paths))
        expected=git(['write-tree'],env).decode().strip()
        git(['read-tree','--empty'],env);stage(paths,env);observed=git(['write-tree'],env).decode().strip()
    assert expected==observed;preserve()
    write(ROOT/'state/private_staging_tree_proof.json',dict(passed=True,expected_tree=expected,observed_tree=observed))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',action='store_true');parser.add_argument('--push',action='store_true');parser.add_argument('--selftest',action='store_true');args=parser.parse_args()
    if args.selftest:proof();return
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists();preserve()
    if args.archive:archive();return
    before=preservation();paths=prepare();parent=git(['rev-parse',BRANCH]).decode().strip()
    assert parent==BASE or git(['merge-base',BASE,parent]).decode().strip()==BASE
    chunks=[];current=[];size=0
    for path in sorted(paths,key=lambda p:('large_evidence.tar.zst.part' in p.name,str(p))):
        if current and size+path.stat().st_size>256*1024**2:chunks.append(current);current=[];size=0
        current.append(path);size+=path.stat().st_size
    if current:chunks.append(current)
    progress=[]
    with tempfile.TemporaryDirectory(prefix='layer_token_private_axis_rootcause_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'));git(['read-tree',parent],env)
        for index,chunk in enumerate(chunks):
            stage(chunk,env);tree=git(['write-tree'],env).decode().strip()
            if tree!=git(['rev-parse',parent+'^{tree}']).decode().strip():
                message=f'BI0 axis causal evidence {index+1}/{len(chunks)}\n\nSeparate queue accounting, GPU measurement and unqualified claims.\n'.encode()
                commit=git(['commit-tree',tree,'-p',parent],env,message).decode().strip()
                git(['update-ref','refs/heads/'+BRANCH,commit,parent]);parent=commit
            if args.push:
                process=subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false','-c','pack.threads=2',
                    'push','origin',parent+':refs/heads/'+BRANCH],cwd=REPO,capture_output=True,text=True)
                progress.append(dict(time=time.time(),checkpoint=index+1,SHA=parent,returncode=process.returncode,stdout=process.stdout,stderr=process.stderr))
                write(ROOT/'state/publication_progress.json',progress);assert process.returncode==0,process.stderr[-3000:]
            print('PUBLICATION_CHECKPOINT',index+1,len(chunks),parent,flush=True)
    remote=git(['ls-remote','origin','refs/heads/'+BRANCH]).decode().split();remote_sha=remote[0] if remote else None
    if args.push:assert remote_sha==parent
    preserve();assert preservation()==before
    write(ROOT/'state/publication_receipt.json',dict(time=time.time(),branch=BRANCH,SHA=parent,Remote_SHA=remote_sha,
        pushed=args.push,remote_matches=remote_sha==parent,user_checkout_and_index_preserved=True))
    print(json.dumps(dict(branch=BRANCH,SHA=parent,Remote_SHA=remote_sha)),flush=True)


if __name__=='__main__':main()
