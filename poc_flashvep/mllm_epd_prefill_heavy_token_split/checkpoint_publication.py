"""Publish an explicit incomplete checkpoint without touching the user index."""
import gzip
import hashlib
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,digest,preserve,write


def raw_records():
    folder=ROOT/'archives/public_checkpoint_v0';folder.mkdir(exist_ok=True)
    manifest=ROOT/'archives/checkpoint_raw_records_v0.json'
    if manifest.exists():
        value=json.loads(manifest.read_text())
        assert all(digest(REPO/r['compressed_path'])==r['compressed_SHA256'] for r in value['files'])
        return value
    records=[]
    for source in sorted((ROOT/'raw').glob('*.jsonl')):
        before=digest(source);target=folder/(source.name+'.gz')
        assert not target.exists()
        with source.open('rb') as src,target.open('xb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as dst:
                while block:=src.read(8*1024**2):dst.write(block)
        sha=hashlib.sha256();size=0
        with gzip.open(target,'rb') as stream:
            while block:=stream.read(8*1024**2):sha.update(block);size+=len(block)
        assert sha.hexdigest()==before==digest(source) and size==source.stat().st_size
        records.append(dict(source=str(source.relative_to(REPO)),source_bytes=size,source_SHA256=before,
            compressed_path=str(target.relative_to(REPO)),compressed_bytes=target.stat().st_size,
            compressed_SHA256=digest(target),lossless_verification='PASS'))
    value=dict(time=time.time(),passed=True,files=records,original_raw_preserved=True,
        scope='Checkpoint archive of top-level JSONL request,proxy,host and NVML records. '
            'Nested GPU profiles,tensor captures,media and full worker logs remain local. This is not the final evidence archive.')
    write(manifest,value);return value


def main():
    preserve()
    verification=json.loads((ROOT/'state/main100_subset_verification.json').read_text())
    assert verification['status']=='PASS'
    cpu=ROOT/'logs/cpu_frontend_and_evidence_v0.log'
    assert 'Ran 35 tests' in cpu.read_text() and cpu.read_text().rstrip().endswith('OK')
    write(ROOT/'state/checkpoint_cpu_verification_v0.json',dict(time=time.time(),passed=True,tests=35,
        log=str(cpu),log_SHA256=digest(cpu),scope='CPU protocol,metadata,renderer configuration and evidence-writing regressions. '
            'Not the full Native EPD task-quality or Generic Token Split gate.'))
    archive=raw_records()
    from .source_snapshot import snapshot
    source=snapshot('publication_checkpoint_v0')
    readme=ROOT/'README.md'
    content='\n'.join([
        '# MLLM EPD Main100 Research Checkpoint','',
        'Status: ACTIVE. This is an incomplete research checkpoint, not a final serving result.',
        'The user reduced future main experiments from Main400 to a frozen representative Main100.',
        'The original Main400 and raw results remain unchanged. The subset does not preserve every tail statistic.',
        'The model is Qwen/Qwen2.5-VL-7B-Instruct, original BF16, revision cc594898137f460bfe9f0759e9844b3ce807cfb5.',
        'Main100 SHA256: 910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1.','',
        '## Evidence limits','',
        '- Current serving controls use actual online encoding and installed vLLM EC/NIXL paths.',
        '- ECExampleConnector uses CPU staging through per-run tmpfs. It is a debug connector, not production NIXL encoder transfer.',
        '- Processor-result handoff is a common exact runtime repair. It is not Generic Token Split.',
        '- Current Main100 timing cells are single-start diagnostics. No independent-start serving confidence interval is established.',
        '- Full task quality and cross-geometry encoder gates remain unresolved. Complete source scorers match the latest reference point scores.',
        '- Current physical GPU activity does not confirm P-prefill-heavy or safe additional prefill admission.',
        '- Generic Token Split, inside-image split, online idle-aware split and E/D-safe extra prefill remain NOT_RUN or NOT_MEASURED.',
        '- The CUDA IPC probe tests protocol correctness only. It is not an EC serving connector or an online speedup.',
        '- Future work must preserve the strong baseline, original dataset, output and SLO contracts.','',
        '## Reproduction','',
        'The source module contains dataset,server,client,metric and figure generators.',
        'Configs preserve commands,source hashes,model revision,arrival seeds and GPU identities.',
        'The active manifest is configs/active_workload.json. The archived Main400 remains in data/.',
        'archives/checkpoint_raw_records_v0.json lists lossless compressed top-level JSONL records.',
        'Nested GPU profiles,tensor captures,images and full worker logs remain local at this checkpoint.',
        'The final campaign must archive and verify remaining evidence,finish the gated scope and verify resource/Git closeout.',
        'Source snapshots preserve exact Python bytes, including boot versions where available.',
        'Installed Python wheel integrity does not establish native-library or upstream checkout equivalence.','',
        '## Resource state','',
        'Completed model and protocol jobs released their owned GPUs at safe checkpoints.',
        'Approved burners run while GPUs are unused. This is an inter-phase restoration, not final campaign closeout.',''])
    if readme.exists():assert readme.read_text()==content,'Do not overwrite a different checkpoint README'
    else:readme.write_text(content)
    paths=list(SOURCE.glob('*.py'))
    for pattern in ['mllm_epd*.md','mllm_prefill_heavy*.md']:paths.extend(ROOT.parent.glob(pattern))
    for directory,patterns in [('configs',['*.json']),('state',['*.json']),('analysis',['*.json','*.csv']),
                               ('figures',['*.png','*.pdf']),('data',['*.json'])]:
        for pattern in patterns:paths.extend((ROOT/directory).glob(pattern))
    for pattern in ['*.source.zip']:paths.extend((ROOT/'archives').glob(pattern))
    paths.extend(REPO/r['compressed_path'] for r in archive['files'])
    paths.extend([ROOT/'archives/checkpoint_raw_records_v0.json',readme])
    for directory in ['quality_sources','kernel_only_collector_v3']:
        for pattern in ['*.json','*.cpp','*.cu','*.h']:paths.extend((ROOT/'references'/directory).glob(pattern))
    # Publish public-source provenance only. Do not copy full Notion page exports.
    for name in ['installed_wheel_source_integrity.json']:
        path=ROOT/'references'/name
        if path.exists():paths.append(path)
    paths=sorted(set(paths))
    assert all(p.is_file() for p in paths)
    local=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p),
        reason='Large derived analysis remains local. Its source and lossless top-level raw records are preserved.')
        for p in paths if p.stat().st_size>=90*1024**2]
    assert all('/analysis/' in p['path'] for p in local),'Do not omit a required raw archive or source artifact'
    paths=[p for p in paths if p.stat().st_size<90*1024**2]
    assert all(SOURCE in p.parents or ROOT in p.parents or p.parent==ROOT.parent and p.name.startswith('mllm_') for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v0.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        total_bytes=sum(p.stat().st_size for p in paths),source_archive_SHA256=source['archive_SHA256'],
        local_only_derived_files=local,
        final_goal_complete=False,full_raw_archive_complete=False,model_or_media_weights_published=False,
        user_checkout_and_staged_index='Preserved through a private Git index'))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    value=helper.commit_paths(paths,'WIP MLLM EPD Main100: preserve exact runtime controls and negative headroom diagnostics',
        'publication_commit_checkpoint_v0.json')
    assert value['remote_verified'] and value['passed']
    preserve();print('MLLM_CHECKPOINT_REMOTE_VERIFIED',value['remote_SHA'])


if __name__=='__main__':main()
