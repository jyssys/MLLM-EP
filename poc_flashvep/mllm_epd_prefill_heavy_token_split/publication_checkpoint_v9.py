"""Publish closed first probe block and provenance, not unfinished N3 claims."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,preserve,write,digest
from .publication_checkpoint_v4 import compress

TAGS=('interference_PROBE_C40R4_PREFLIGHT','interference_PROBE_C40R4_B0')
DERIVED=TAGS+('dataset_integrity_','batch_counter_')


def main():
    preserve();test=ROOT/'logs/cpu_regression_v74.log'
    assert 'Ran 211 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    probe=ROOT/'analysis/interference_PROBE_C40R4_PREFLIGHT_summary.json'
    proof=json.loads(probe.read_text());assert proof['status']=='CORRECTNESS_PASS' and proof['prefix_cases']==4
    first=ROOT/'analysis/interference_PROBE_C40R4_B0_summary.json'
    assert json.loads(first.read_text())['status']=='COMPLETE'
    integrity=ROOT/'analysis/dataset_integrity_audit_v0.json';data=json.loads(integrity.read_text())
    assert data['passed'] and data['requests']==432 and data['unique_media']==3134
    assert data['model_files']==15 and data['subset_unchanged']
    closed=[];source_archives=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAGS):continue
        row=json.loads(path.read_text());assert row['finished'] and not Path('/proc',str(row['guardian'])).exists()
        closed.append(path);archive=Path(row['source_archive'])
        assert digest(archive)==row['source_archive_SHA256'];source_archives.append(archive)
    assert len(closed)==11
    raw={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith(TAGS+tuple('proxy_'+tag for tag in TAGS))}
    folder=ROOT/'archives/public_checkpoint_v9';manifest=ROOT/'archives/checkpoint_raw_records_v9.json'
    assert not folder.exists() and not manifest.exists();folder.mkdir()
    compressed=[compress(path,folder) for path in sorted(raw)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed probe preflight and B0 original responses, lossless gzip with round-trip SHA256. '
            'Captured tensors and original media stay local with recorded hashes. B1/B2, suffix, Native ABBA and counter GPU results are excluded.'))
    readme=ROOT/'README_checkpoint_v9.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v9','',
        'Status: ACTIVE and INCOMPLETE. No final resource closeout or goal completion.',
        'The corrected E/D probe preflight passed all28 KV layers for128/512-token prefixes on each helper.',
        'The first independent interference block completed. Its eight probe cohorts accepted20–32 real prefix probes each.',
        'These probes discard their sample and do not complete a Main100 task. They do not increase the client throughput numerator.',
        'The remaining independent blocks are live. No N3 protection or safe-headroom verdict is inferred from this checkpoint.',
        'The frozen Main400, Main100 and Calibration32 content and context split were reverified.',
        'All432 prompt-token/MRoPE metadata records,3134 unique original images,37 used source parquet files,87 PDFs and15 model files passed byte checks.',
        'Image pixel hashes passed. The model files match the original pinned download receipts.',
        'Frozen structural encoder output shapes remain structural. This audit does not invent per-request GPU shape measurements.',
        'The latest returned Notion direction keeps Main100, excludes video and rejects the historical4x4 quota.',
        'A separate contract audit discloses that conservative Token C3 ran before the full prefill-heavy gate passed.',
        'The reported C3 runtime failures are not a success confirmation in the requested prefill-heavy regime.',
        'New primary-paper notes cover RServe and A Matter of Stages. Raincheck remains withdrawn in the latest lookup.',
        'Only paper provenance metadata is published here. Original paper PDFs and extracted full text remain local.',
        'An inactive counter-only observer records prepared prompt/generation query rows and encoder output shapes without CUDA events or tensor copies.',
        'Its bounded CPC experiment is frozen and waits for the earlier owned jobs to close. No counter GPU result is claimed.',
        'Original model mathematics, compiled baselines and candidate admission remain unchanged. CPU regression:211PASS.',
        'Full task quality and actual high-load E/D-safe headroom remain unresolved.','']))
    paths=list(SOURCE.glob('*.py'))+closed+source_archives+[test,probe,first,integrity,manifest,readme,
        ROOT.parent/'mllm_epd_campaign_contract_and_deviations_v1.md',
        ROOT.parent/'mllm_epd_related_work_final_supplement_v1.md',
        ROOT/'configs/batch_COUNTER_C46.json']
    for directory in ['analysis','configs','state','logs']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<80*1024**2:paths.append(path)
    for name in ['notion_final_direction_refresh_v0.json','notion_final_archive_refresh_v0.json','notion_final_archive_query_v0.json']:
        paths.append(ROOT/'references'/name)
    for name in ['rserve_supplement_v1.json','a_matter_stages_supplement_v1.json']:
        paths.append(ROOT/'references/primary_papers'/name)
    for name in ['cpu_regression_v71.log','cpu_regression_v72.log','cpu_regression_v73.log','related_work_final_supplement_v0.log']:
        paths.append(ROOT/'logs'/name)
    for path in (ROOT/'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith(TAGS):paths.append(path)
    paths.extend(REPO/record['compressed_path'] for record in compressed)
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v9.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        bytes=sum(p.stat().st_size for p in paths),closed_GPU_leases=len(closed),
        probe_N3='LIVE_ONLY_PREFLIGHT_AND_B0_PUBLISHED',counter_GPU='WAITING_AND_NOT_RUN',
        task_quality='PENDING',final_goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,'WIP MLLM EPD: actual E/D probe preflight, frozen provenance and gated batch-counter protocol',
        'publication_commit_checkpoint_v9.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V9_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
