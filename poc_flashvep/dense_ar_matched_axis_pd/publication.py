"""Reuse private-index publication without changing the user checkout."""
import json
import collections
import gzip
from .common import BASE,BRANCH,REPO,ROOT,SOURCE,digest,preserve,preservation,write

DOCUMENTS=('dense_ar_matched_axis_crossover_spec.md',
    'dense_ar_matched_axis_crossover_final_summary.md',
    'dense_ar_unified_b8192_s1024_runtime_audit.md',
    'dense_ar_matched_work_output_sweep_results.md',
    'dense_ar_layer_token_crossover_root_cause.md',
    'dense_ar_unified_budget_duetserve_motivation_results.md',
    'dense_ar_matched_axis_output_sweep_figures.md',
    'dense_ar_last2_last4_actual_suffix_quick_comparison.md',
    'dense_ar_last2_matched_sweep_supplement_scope.md',
    'dense_ar_matched_axis_small_start_ci_sensitivity.md',
    'dense_ar_matched_axis_daemon_recovery_20261118.md',
    'dense_ar_actual_batch_matched_online_results.md',
    'dense_ar_batch_matched_axis_tradeoff_protocol.md',
    'dense_ar_fixed_matched_cohort2_runtime_and_protocol.md',
    'dense_ar_serving_projection_api_measurement_audit.md',
    'dense_ar_small_cut_gemm_to_ttft_diagnosis_progress.md')


def candidates():
    archive=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    archived={r['path'] for r in archive['original_files']}
    for subfolder in ('analysis_addendum','raw_metadata_addendum'):
        addendum=ROOT/'archives'/subfolder/'large_evidence_manifest.json'
        if addendum.exists():
            extra=json.loads(addendum.read_text())
            assert extra['verified_lossless']
            archived.update(r['path'] for r in extra['original_files'])
    selected=[];excluded=[]
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():continue
        relative=str(path.relative_to(REPO));reason=None
        if path.name in ('publication_coverage.json','publication_SHA256_manifest.json','publication_exclusions.jsonl.gz'):
            continue
        if 'worker_source' in path.parts:
            reason='EXTRACTED_DUPLICATE_OF_FROZEN_SOURCE_ZIP'
        elif '__pycache__' in path.parts or 'tools' in path.relative_to(ROOT).parts:
            reason='REGENERABLE_CACHE_NOT_PUBLISHED'
        elif relative in archived:
            reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif path.suffix=='.pdf' and 'references' in path.relative_to(ROOT).parts:
            reason='EXTERNAL_PAPER_NOT_REDISTRIBUTED'
        elif path.suffix in ('.lock','.local') or '.tmp.' in path.name:
            reason='LOCAL_OPERATIONAL_FILE'
        elif path.name in ('publication_receipt.json','publication_progress.json',
                'push_command_result.json','post_publication_final_check.json'):
            reason='POST_COMMIT_LOCAL_RECEIPT'
        elif path.name=='burner.log':
            reason='LIVE_LOG_WITH_FROZEN_CLOSEOUT_SNAPSHOT'
        elif path.suffix=='.bin' and 'state_transport' in path.parts:
            reason='EXPIRED_REGISTRATION_DESCRIPTOR_HASH_PRESERVED'
        if reason:
            row=dict(path=relative,bytes=path.stat().st_size,reason=reason)
            if reason=='EXPIRED_REGISTRATION_DESCRIPTOR_HASH_PRESERVED':row['SHA256']=digest(path)
            excluded.append(row)
        else:
            assert not path.is_symlink() and path.stat().st_size<100*1024**2,relative
            selected.append(path)
    selected.extend(SOURCE.glob('*.py'))
    selected.extend(REPO/'poc_flashvep/reports'/name for name in DOCUMENTS)
    stems=['dense_ar_unified_four_metrics_o200_last'+str(m) for m in (4,8,12)]
    stems.append('dense_ar_unified_qps_four_metrics_o200')
    stems.extend('dense_ar_batch_matched_four_metrics_o'+str(o) for o in (16,64))
    selected.extend(REPO/'poc_flashvep'/(stem+suffix) for stem in stems for suffix in ('.png','.pdf'))
    final_figures=json.loads((ROOT/'analysis/scoped_final_figure_manifest.json').read_text())
    selected.extend(REPO/r['path'] for r in final_figures['files']
                    if (REPO/r['path']).parent==REPO/'poc_flashvep')
    assert all(path.exists() for path in selected)
    return sorted(set(selected)),excluded


def prepare():
    paths,excluded=candidates()
    ledger=ROOT/'state/publication_exclusions.jsonl.gz'
    with gzip.open(ledger,'wt',encoding='utf-8') as stream:
        for row in excluded:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    coverage=ROOT/'state/publication_coverage.json'
    manifest=ROOT/'state/publication_SHA256_manifest.json'
    payload=sorted(set(paths+[ledger]))
    write(coverage,dict(source_parent=BASE,selected_files=len(payload)+2,
        payload_bytes_excluding_publication_metadata=sum(p.stat().st_size for p in payload),
        excluded_files=len(excluded),excluded_reason_counts=dict(collections.Counter(r['reason'] for r in excluded)),
        exclusions_ledger=str(ledger.relative_to(REPO)),exclusions_ledger_SHA256=digest(ledger),
        verified_archive_manifests=[str(p.relative_to(REPO)) for p in sorted((ROOT/'archives').rglob('large_evidence_manifest.json'))],
        scope='Complete per-file exclusion records are preserved as gzip JSONL to remain below the GitHub file-size limit. All archived originals stay local. No user-index, checkout, reset or force-push operation occurs.'))
    write(manifest,dict(files=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p))
        for p in sorted(payload+[coverage])],self_included=False,
        scope='SHA256 covers published payload and coverage. Each archive manifest covers its decompressed original files. The final commit binds this manifest. Post-commit local receipts and live burner log are excluded explicitly.'))
    result=sorted(set(payload+[coverage,manifest]))
    assert all(p.stat().st_size<100*1024**2 for p in result)
    return result


def main():
    from .common import bind_legacy
    bind_legacy()
    from ..dense_ar_layer_token_pd import publication as proven
    for key,value in dict(BASE=BASE,BRANCH=BRANCH,REPO=REPO,ROOT=ROOT,SOURCE=SOURCE,
            digest=digest,preserve=preserve,preservation=preservation,write=write,
            DOCUMENTS=DOCUMENTS,candidates=candidates,prepare=prepare).items():
        setattr(proven,key,value)
    # The proven publisher builds a temporary index from BASE. It never calls
    # checkout, reset, or update-index on the user's index. Archive parts use
    # verified original-member SHA256 values and remain below100MiB each.
    proven.main()


if __name__=='__main__':
    main()
