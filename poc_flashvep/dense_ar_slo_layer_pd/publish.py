"""Publish campaign files with a private index and verified lossless evidence."""
from .common import *

DOCUMENTS=['dense_ar_slo_throughput_dynamic_layer_pd_spec.md',
    'dense_ar_slo_throughput_dynamic_layer_pd_final_summary.md',
    'dense_ar_slo_specific_colocated_frontier.md',
    'dense_ar_long_output_zero_migration_root_cause.md',
    'dense_ar_deadline_aware_small_cut_admission.md',
    'dense_ar_heterogeneous_bursty_crossover_results.md',
    'dense_ar_sustainable_generation_throughput.md',
    'dense_ar_dynamic_vs_static_matched_budget.md',
    'dense_ar_layer_pd_main_figures_and_statistics.md']


def candidates():
    manifest=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
    archived={r['path'] for r in manifest['original_files']};selected=[];excluded=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        name=str(p.relative_to(REPO));reason=None
        if 'worker_source' in p.parts:reason='EXTRACTED_DUPLICATE_OF_VERIFIED_SOURCE_ZIP'
        elif '__pycache__' in p.parts:reason='REGENERABLE_PYTHON_CACHE'
        elif 'tools' in p.relative_to(ROOT).parts:reason='REGENERABLE_EXTENSION_OR_COMPILE_CACHE_SOURCE_IS_PRESERVED'
        elif name in archived:reason='VERIFIED_LOSSLESS_ARCHIVE'
        elif p.suffix in ('.lock','.local') or '.tmp.' in p.name:reason='LOCAL_OPERATIONAL_FILE'
        elif p.name in ('publication_receipt.json','publication_progress.json'):reason='POST_COMMIT_LOCAL_RECEIPT'
        elif p.name=='burner.log':reason='LIVE_LOG_HAS_CLOSEOUT_SNAPSHOT'
        elif p.suffix=='.bin' and 'state_transport' in p.parts:reason='EXPIRED_REGISTRATION_METADATA_HASHED_IN_DESCRIPTOR'
        if reason:excluded.append(dict(path=name,bytes=p.stat().st_size,reason=reason));continue
        assert not p.is_symlink() and p.stat().st_size<100*1024**2,name
        selected.append(p)
    selected+=sorted(p for p in SOURCE.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and p.suffix in ('.py','.txt','.md'))
    selected+=[REPO/'poc_flashvep/reports'/name for name in DOCUMENTS]
    assert all(p.exists() for p in selected)
    return sorted(set(selected)),excluded


def main():
    from ..dense_ar_dynamic_layer_pd import publish as implementation
    for name in ('ROOT','REPO','SOURCE','BASE','BRANCH','preserve','preservation','digest','write','git'):
        setattr(implementation,name,globals()[name])
    implementation.DOCUMENTS=DOCUMENTS;implementation.candidates=candidates
    implementation.main()


if __name__=='__main__':main()
