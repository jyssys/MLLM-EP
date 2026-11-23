"""Publish only this campaign through a private index and an explicit branch."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from .common import ROOT,REPO,SOURCE,BASE,BRANCH,write,digest,preserve

def git(args,**kwargs):return subprocess.check_output(['git']+args,cwd=REPO,text=True,**kwargs).strip()

def stage_paths(paths,env):
    if not paths:return
    relative=[str(path.relative_to(REPO)) for path in paths]
    assert all('\n' not in name and '\r' not in name and not name.startswith('"') for name in relative)
    blobs=git(['hash-object','-w','--stdin-paths'],input='\n'.join(relative)+'\n',env=env).splitlines()
    assert len(blobs)==len(relative)
    records=''.join(f'100644 {blob}\t{name}\0' for blob,name in zip(blobs,relative))
    git(['update-index','-z','--index-info'],input=records,env=env)

def commit_paths(paths,message,prefix):
    began=time.time()
    preserve();parent=git(['rev-parse',f'refs/heads/{BRANCH}'])
    assert parent==BASE or (ROOT/'state/publication_checkpoint_v2.json').exists(),'Unexpected branch parent'
    paths=sorted(set(p.resolve() for p in paths));assert all(REPO in p.parents and p.is_file() for p in paths)
    assert all(p.stat().st_size<90*1024**2 for p in paths),'Split large evidence artifacts before publication'
    with tempfile.TemporaryDirectory(prefix='longbench400-v2-private-index-') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'))
        git(['read-tree',parent],env=env)
        stage_paths(paths,env)
        tree=git(['write-tree'],env=env)
        process=subprocess.run(['git','commit-tree',tree,'-p',parent],input=message+'\n',cwd=REPO,env=env,text=True,capture_output=True,check=True)
        commit=process.stdout.strip();git(['update-ref',f'refs/heads/{BRANCH}',commit,parent])
    preserve();subprocess.run(['git','push','--porcelain','origin',f'refs/heads/{BRANCH}:refs/heads/{BRANCH}'],cwd=REPO,check=True)
    remote=git(['ls-remote','origin',f'refs/heads/{BRANCH}']).split()[0];assert remote==commit
    result=dict(time=time.time(),elapsed_seconds=time.time()-began,branch=BRANCH,parent=parent,commit=commit,remote_SHA=remote,remote_verified=True,
        files=[dict(path=str(p.relative_to(REPO)),SHA256=digest(p)) for p in paths],user_checkout_and_index_preserved=True)
    write(ROOT/'state'/f'{prefix}.json',result);preserve();print('PUBLISHED_AND_REMOTE_VERIFIED',commit,flush=True);return result

def checkpoint():
    assert not (ROOT/'state/publication_checkpoint_v2.json').exists()
    files=list(SOURCE.glob('*.py'))
    reports=('dense_ar_longbench400_v2_checkpoint.md','dense_ar_longbench400_v2_dataset_contract.md',
        'dense_ar_longbench400_v2_candidate_availability.md')
    files += [REPO/'poc_flashvep/reports'/name for name in reports]
    files += [ROOT/'data/balanced400_v2_freeze_manifest.json',ROOT/'configs/dataset_selection_contract_v2.json',
        ROOT/'analysis/post_expansion_candidate_availability.json',ROOT/'analysis/controller_dependency_base_audit_v2.json',
        ROOT/'references/official_longbench_dataset_card_v1.json',ROOT/'references/longbench_code_license_mit.json']
    files += list((ROOT/'references/source_capsules').glob('*.zip'))+list((ROOT/'references/source_capsules').glob('*.json'))
    return commit_paths(files,'LongBench400-v2 checkpoint: freeze dataset and qualify compiler repairs; research remains active','publication_checkpoint_v2')

def c0_checkpoint():
    assert json.loads((ROOT/'state/c0_campaign_complete_v2.json').read_text())['complete']
    assert not (ROOT/'state/publication_c0_checkpoint_v5.json').exists()
    files=list(SOURCE.glob('*.py'))
    files+=list((REPO/'poc_flashvep/reports').glob('dense_ar_longbench400_v2*.md'))
    files+=list((ROOT/'figures').glob('c0_*'))+list((ROOT/'figures').glob('v2_dataset_*'))
    files+=list((ROOT/'figures').glob('pd_attention_and_payload_*'))
    files+=list((ROOT/'publication/provenance').glob('*.json'))
    for pattern in ('c0_output*','c0_controlled*','c0_observable*','heldout_output_prior*',
                    'pd_attention_and_payload*','model_gemm_dimension*','stock_*audit*'):
        files+=list((ROOT/'analysis').glob(pattern+'.json'))
    for folder in ('configs','state'):
        for pattern in ('c0_output_selector*','c3_*budget*','c3_*priority*','c3_fused_teacher_context*',
                        'c0_campaign_complete*','c0_phase_analysis_complete*'):
            files+=list((ROOT/folder).glob(pattern+'.json'))
    files+=list((ROOT/'references/source_capsules').glob('*.zip'))
    files+=list((ROOT/'references/source_capsules').glob('*.json'))
    return commit_paths(files,'LongBench400-v2 active checkpoint: complete N3 C0 output map; retain safety failures and C3 work',
                        'publication_c0_checkpoint_v5')

def final():
    assert json.loads((ROOT/'state/resource_closeout_v2.json').read_text())['passed']
    assert json.loads((ROOT/'state/report_generation_v2.json').read_text())['final']
    assert json.loads((ROOT/'analysis/raw_metric_recalculation_v2.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/final_configuration_and_gate_inventory_v2.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/lossless_archive_integrity_v2.json').read_text())['passed']
    files=list(SOURCE.glob('*.py'))
    files+=list((REPO/'poc_flashvep/reports').glob('dense_ar_longbench400_v2*.md'))
    for directory,patterns in (('figures',('*.png','*.pdf','*.csv')),('configs',('*.json',)),
            ('jobs',('*.json',)),('analysis',('*.json',)),('state',('*.json',)),
            ('local_reconstruction_validation',('reconstruction_validation_v2.json',)),('local_dataset_archives',('*manifest*.json',)),
            ('local_tensor_archives',('*.json',))):
        for pattern in patterns:files+=list((ROOT/directory).glob(pattern))
    files+=list((ROOT/'references').glob('*.json'))
    files+=list((ROOT/'references/source_capsules').glob('*.zip'))
    files+=list((ROOT/'references/source_capsules').glob('*.json'))
    for directory in ('kernel_only_collector_v3','correlated_collector_v1'):
        for pattern in ('*.json','*.cpp','*.cu','*.h'):
            files+=list((ROOT/'references'/directory).glob(pattern))
    for directory in ('publication/provenance','publication/minimal_evidence/manifests',
                      'publication/evidence/manifests','publication/evidence/sources'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix not in ('.pt','.bin','.so'):files.append(path)
    for path in (ROOT/'publication/minimal_evidence/manifests').glob('*.json'):
        files += [ROOT/record['path'] for record in json.loads(path.read_text())['compressed_parts']]
    for path in (ROOT/'publication/derived_tables').glob('*.json'):
        files.append(path)
        record=json.loads(path.read_text())
        files += [ROOT/part['path'] for part in record.get('compressed_parts',[])]
    # Keep all bulk dataset/context and large tensor archives local. The public
    # identity rosters,official-source hashes and reconstruction code are enough
    # to recover inputs under their original source licences.
    selected=[];local=[]
    for path in sorted(set(files)):
        if path.stat().st_size>=90*1024**2:
            local.append(dict(path=str(path.relative_to(REPO)),bytes=path.stat().st_size,SHA256=digest(path),
                reason='The original derived artifact remains local. A lossless archive is published when listed in publication/derived_tables/index_v2.json. Inputs and regeneration code are also published.'))
        else:selected.append(path)
    write(ROOT/'state/publication_local_only_derived_files_v2.json',dict(time=time.time(),files=local,
        bulk_context_and_tensor_archives_published=False,
        scope='Full raw logs and tensor archives remain local with SHA256 manifests. Publish structured run archives and exact sources.'))
    selected.append(ROOT/'state/publication_local_only_derived_files_v2.json')
    return commit_paths(selected,'LongBench400-v2: report output-aware C0 and strong C3 results,compiler controls and decode gates',
                        'publication_final_v2')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('checkpoint','c0_checkpoint','final'));args=p.parse_args();globals()[args.action]()
