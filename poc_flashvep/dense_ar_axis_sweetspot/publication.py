"""Reuse the tested private-index publisher for this namespace only."""
import collections
import gzip
import json
from . import common
from .common import ROOT,SOURCE,REPO,write,digest

DOCUMENTS=('dense_ar_axis_sweetspot_knob_discovery_final_summary.md',
    'dense_ar_axis_existing_evidence_root_cause_audit.md',
    'dense_ar_axis_batch_matched_online_results.md',
    'dense_ar_axis_service_queue_causal_decomposition.md',
    'dense_ar_axis_minimal_knob_discovery.md',
    'dense_ar_axis_offline_oracle_headroom.md',
    'dense_ar_axis_sweetspot_main_figures.md',
    'dense_ar_axis_research_decision_log.md')


def candidates():
    manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    assert manifest['verified_lossless']
    archived={row['path'] for row in manifest['original_files']}
    selected=[];excluded=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=str(path.relative_to(REPO));reason=None
        if path.name in ('publication_coverage.json','publication_SHA256_manifest.json','publication_exclusions.jsonl.gz'):
            continue
        if 'worker_source' in path.parts:reason='EXTRACTED_DUPLICATE_OF_VERIFIED_SOURCE_ZIP'
        elif '__pycache__' in path.parts or 'tools' in path.relative_to(ROOT).parts:reason='REGENERABLE_CACHE'
        elif relative in archived:reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif path.suffix=='.pdf' and 'references' in path.relative_to(ROOT).parts:reason='EXTERNAL_PAPER_NOT_REDISTRIBUTED'
        elif path.suffix in ('.lock','.local') or '.tmp.' in path.name:reason='LOCAL_OPERATIONAL_FILE'
        elif path.name in ('publication_receipt.json','publication_progress.json','push_command_result.json','post_publication_final_check.json'):
            reason='POST_COMMIT_LOCAL_RECEIPT'
        elif path.name=='burner.log':reason='LIVE_LOG_WITH_FROZEN_CLOSEOUT_SNAPSHOT'
        elif path.suffix=='.bin' and 'state_transport' in path.parts:reason='EXPIRED_REGISTRATION_DESCRIPTOR_HASH_ONLY'
        if reason:
            item=dict(path=relative,bytes=path.stat().st_size,reason=reason)
            if reason=='EXPIRED_REGISTRATION_DESCRIPTOR_HASH_ONLY':item['SHA256']=digest(path)
            excluded.append(item)
        else:
            assert not path.is_symlink() and path.stat().st_size<100*1024**2,relative
            selected.append(path)
    selected.extend(SOURCE.glob('*.py'))
    selected.extend(REPO/'poc_flashvep/reports'/name for name in DOCUMENTS)
    selected.extend((REPO/'poc_flashvep').glob('dense_ar_axis_sweetspot_four_metrics_o200_last*.png'))
    selected.extend((REPO/'poc_flashvep').glob('dense_ar_axis_sweetspot_four_metrics_o200_last*.pdf'))
    assert all(p.exists() for p in selected)
    return sorted(set(selected)),excluded


def prepare():
    selected,excluded=candidates()
    ledger=ROOT/'state/publication_exclusions.jsonl.gz'
    with gzip.open(ledger,'wt',encoding='utf-8') as stream:
        for row in excluded:stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    coverage=ROOT/'state/publication_coverage.json';manifest=ROOT/'state/publication_SHA256_manifest.json'
    payload=sorted(set(selected+[ledger]))
    write(coverage,dict(source_parent=common.BASE,selected_files=len(payload)+2,
        payload_bytes_excluding_publication_metadata=sum(p.stat().st_size for p in payload),
        excluded_files=len(excluded),excluded_reason_counts=dict(collections.Counter(r['reason'] for r in excluded)),
        exclusions_ledger=str(ledger.relative_to(REPO)),exclusions_SHA256=digest(ledger),
        scope='New source, reports, figures and evidence only. Lossless archive originals stay local. Preserve the user index, checkout and external runtime.'))
    write(manifest,dict(files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p))
        for p in sorted(payload+[coverage])],self_included=False,
        scope='Each published file has SHA256. Archive members have separate verified hashes. The Git commit binds this manifest. Live burner log and post-commit receipts are excluded explicitly.'))
    return sorted(set(payload+[coverage,manifest]))


def main():
    common.bind_legacy()
    from ..dense_ar_layer_token_pd import publication as proven
    for key in ('BASE','BRANCH','REPO','ROOT','SOURCE','digest','preserve','preservation','write'):
        setattr(proven,key,getattr(common,key))
    proven.DOCUMENTS=DOCUMENTS
    proven.candidates=candidates;proven.prepare=prepare
    proven.main()


if __name__=='__main__':
    main()
