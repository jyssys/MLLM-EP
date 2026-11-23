"""Publish the scoped campaign through a private index without checkout changes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from .common import ROOT,REPO,SOURCE,BASE,BRANCH,write,digest,preserve
from .reports import NAMES


def git(args,**kwargs):return subprocess.check_output(['git']+args,cwd=REPO,text=True,**kwargs).strip()


def selected_paths():
    required=('analysis/cpu_regression_audit.json','analysis/raw_metric_recalculation.json',
        'analysis/configuration_and_gate_inventory.json','state/science_completion_audit.json',
        'state/resource_closeout.json','archives/evidence_archive_manifest.json')
    for relative in required:
        value=json.loads((ROOT/relative).read_text());assert value['passed'],relative
    archive=json.loads((ROOT/'archives/evidence_archive_manifest.json').read_text())
    paths=list(SOURCE.glob('*.py'))+[ROOT.parent/name for name in NAMES]
    if (ROOT.parent/'dense_ar_attention_head_offloading_poc.md').exists():paths.append(ROOT.parent/'dense_ar_attention_head_offloading_poc.md')
    for directory,patterns in (('configs',('*.json',)),('jobs',('*.json',)),('state',('*.json','*.jsonl')),
        ('analysis',('*.json',)),('figures',('*.png','*.pdf','*.csv')),('references',('*.json',)),
        ('references/source_capsules',('*.zip','*.json')),('publication/provenance',('*.json',))):
        for pattern in patterns:paths.extend((ROOT/directory).glob(pattern))
    for relative in ('references/kernel_only_collector_v3','references/correlated_collector_v1'):
        for pattern in ('*.cpp','*.cu','*.h','*.json'):paths.extend((ROOT/relative).glob(pattern))
    for directory in ('configs','analysis','state','references','figures'):
        for extension in ('json','zip','csv','png','pdf'):paths.extend((ROOT/'attention_poc'/directory).glob('*.'+extension))
    paths.append(ROOT/'archives/evidence_archive_manifest.json')
    paths.extend(REPO/part['path'] for part in archive['public_archive']['parts'])
    if (ROOT/'README.md').exists():paths.append(ROOT/'README.md')
    local=[];chosen=[]
    for path in sorted(set(paths)):
        if path.stat().st_size>=90*1024**2:
            local.append(dict(path=str(path.relative_to(REPO)),bytes=path.stat().st_size,SHA256=digest(path),
                reason='Large derived file stays local. Published raw evidence and source code reproduce it.'))
        else:chosen.append(path)
    record=ROOT/'state/publication_selection.json'
    write(record,dict(time=time.time(),selected=len(chosen),local_only_derived=local,
        third_party_contexts_published=False,model_weights_published=False,
        scope='Publish only the new campaign source,reports,structured evidence and verified public archive parts. '
            'The user checkout,index and other branches are not changed. Large local archives remain recoverable by manifest.'))
    chosen.append(record)
    return chosen


def commit_paths(paths,message,record_name):
    preserve();parent=git(['rev-parse','refs/heads/'+BRANCH])
    if parent!=BASE:
        records=list((ROOT/'state').glob('publication_commit_*.json'))
        assert any(json.loads(path.read_text())['commit']==parent for path in records),'Unexpected research branch parent'
    paths=sorted(set(path.resolve() for path in paths))
    assert all(REPO in path.parents and path.is_file() and path.stat().st_size<90*1024**2 for path in paths)
    names=[str(path.relative_to(REPO)) for path in paths]
    assert all('\n' not in name and '\r' not in name and not name.startswith('"') for name in names)
    with tempfile.TemporaryDirectory(prefix='p_critical_private_index_') as folder:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(folder)/'index'))
        git(['read-tree',parent],env=env)
        blobs=git(['hash-object','-w','--stdin-paths'],input='\n'.join(names)+'\n',env=env).splitlines();assert len(blobs)==len(names)
        payload=''.join(f'100644 {blob}\t{name}\0' for blob,name in zip(blobs,names))
        git(['update-index','-z','--index-info'],input=payload,env=env)
        tree=git(['write-tree'],env=env)
        commit=git(['commit-tree',tree,'-p',parent],input=message+'\n',env=env)
        git(['update-ref','refs/heads/'+BRANCH,commit,parent])
    preserve()
    value=dict(time=time.time(),passed=False,branch=BRANCH,parent=parent,commit=commit,remote_SHA=None,
        remote_verified=False,user_checkout_and_index_preserved=True,phase='LOCAL_COMMIT_CREATED',
        files=[dict(path=str(path.relative_to(REPO)),SHA256=digest(path)) for path in paths])
    write(ROOT/'state'/record_name,value);preserve()
    return push_record(record_name)


def push_record(record_name):
    preserve();path=ROOT/'state'/record_name;value=json.loads(path.read_text());commit=value['commit']
    assert value['branch']==BRANCH and git(['rev-parse','refs/heads/'+BRANCH])==commit
    assert all(digest(REPO/row['path'])==row['SHA256'] for row in value['files']), 'Recorded files changed after the local commit'
    subprocess.run(['git','push','--porcelain','origin',f'refs/heads/{BRANCH}:refs/heads/{BRANCH}'],cwd=REPO,check=True)
    remote=git(['ls-remote','origin','refs/heads/'+BRANCH]).split()[0];assert remote==commit
    value.update(time=time.time(),passed=True,phase='REMOTE_SHA_VERIFIED',remote_SHA=remote,remote_verified=True)
    write(path,value);preserve()
    print('P_CRITICAL_PUBLISHED_REMOTE_VERIFIED',commit,flush=True);return value


def publish():
    existing=ROOT/'state/publication_commit_science.json'
    if existing.exists() and not json.loads(existing.read_text())['passed']:
        return push_record(existing.name)
    paths=selected_paths()
    return commit_paths(paths,'Dense AR P-critical-path PoC: preserve matched replay,numerical failures and decode gates',
        'publication_commit_science.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selection-only',action='store_true');args=parser.parse_args()
    print(len(selected_paths())) if args.selection_only else publish()
