"""Publish suffix diagnostics and the completed Native EC policy intervention."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,preserve,write,digest
from .publication_checkpoint_v4 import compress

TAGS=('suffix_QUANTA_C41R3','ec_INLINE_CAUSAL_C44')
DERIVED=TAGS+('dual_side_headroom_','idle_addendum_final_','probe_reference_','gpu_counter_capability_',
    'delivery_requirement_inventory_v3','delivery_requirement_figures_v3')
REPORTS=['mllm_epd_imported_kv_suffix_quantum_cost_v1.md','mllm_epd_native_ec_descriptor_causal_control_v1.md',
    'mllm_epd_dual_side_prefill_headroom_report.md','mllm_epd_encoder_decoder_idle_final_summary.md',
    'mllm_epd_probe_reference_policy_limits_v1.md']


def main():
    preserve();test=ROOT/'logs/cpu_regression_v87.log'
    assert 'Ran 238 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    suffix=ROOT/'analysis/suffix_QUANTA_C41R3_report_audit_v0.json';cost=json.loads(suffix.read_text())
    assert cost['query_work_conserved'] and cost['evidence']=='DIAGNOSTIC' and not cost['observer_gate']
    policy=ROOT/'analysis/ec_INLINE_CAUSAL_C44_audit_v0.json';control=json.loads(policy.read_text())
    assert control['cohorts']==24 and control['independent_initializations']==3 and control['metric_recalculation']=='PASS'
    inventory=ROOT/'analysis/delivery_requirement_inventory_v3.json';items=json.loads(inventory.read_text())
    assert sum(r['artifact_available'] for r in items['figures'])==26
    assert sum(r['available'] for r in items['reports'])==13 and not items['goal_complete']
    assert not (ROOT/'state/final_resource_closeout_v0.json').exists(),'This is an inter-phase checkpoint, not final closeout.'
    closed=[];archives=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAGS):continue
        row=json.loads(path.read_text());assert row['finished'] and not Path('/proc',str(row['guardian'])).exists()
        closed.append(path);archive=Path(row['source_archive'])
        assert digest(archive)==row['source_archive_SHA256'];archives.append(archive)
    assert len(closed)==12
    raw={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith(TAGS+tuple('proxy_'+tag for tag in TAGS)+tuple('NVML_'+tag for tag in TAGS))}
    folder=ROOT/'raw/gpu_enclosures/suffix_QUANTA_C41R3'
    raw.update(path for path in folder.rglob('*') if path.is_file() and path.suffix in ('.csv','.json'))
    target=ROOT/'archives/public_checkpoint_v12';manifest=ROOT/'archives/checkpoint_raw_records_v12.json'
    assert not target.exists() and not manifest.exists();target.mkdir()
    compressed=[compress(path,target) for path in sorted(raw)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed imported-KV suffix and Native policy responses plus forward-enclosure records. '
            'Lossless gzip round-trip hashes. Shared append-only EC logs and live counter results are excluded. '
            'Large original tensor payloads stay local with their original checksum receipts.'))
    readme=ROOT/'README_checkpoint_v12.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v12','',
        'Status: ACTIVE and INCOMPLETE. Actual batch validation is still live.',
        'All15 imported-prefix suffix correctness cases passed. Actual D query work was conserved.',
        'The suffix cost observer gate failed. Every cost and physical-gap fit retains its diagnostic or offline scope.',
        'The Native EC policy intervention completed24 cohorts across three independent initializations.',
        'Two periods per policy were averaged within each initialization before paired N3 inference.',
        'Inline descriptors reduced TTFT relative to On-demand in this Native control.',
        'That result does not repair Stock Token or NativeStrong, which already used Inline descriptors.',
        'The policy contrast did not pass every relative decode gate. It is not a new token-split serving GO.',
        'Host-stage differences are not GPU-exclusive compute, DMA or an additive p95 decomposition.',
        'The frozen E/D idle addendum reports low-load physical overlap, not confirmed high-load safe capacity.',
        'The no-probe reference audit preserves baseline variation. Not every rare client gap is uniquely attributed to a probe.',
        'All26 requested figure items have corresponding artifacts. Artifact presence does not establish scientific PASS.',
        'Thirteen of14 named reports exist. The overall campaign final summary remains pending.',
        'Final figure reproduction uses fresh output folders and blocks Python writes through links into original evidence.',
        'It has not run during active serving. PDF checks do not claim unavailable pixel rendering.',
        'The expensive original-tensor, gzip and source-archive recheck is also deferred until actual resource closeout.',
        'Read-only counter capability facts record administrator-only profiling and no effective capabilities.',
        'No driver setting, privilege or active measurement session was changed.',
        'CPU regression:238PASS. Original images, model weights, SLOs, checkout and index remain protected.',
        'Full task quality, actual high-load safe headroom, final resource restoration and final publication remain unresolved.','']))
    paths=list(SOURCE.glob('*.py'))+closed+archives+[test,suffix,policy,inventory,manifest,readme]
    paths.extend(ROOT.parent/name for name in REPORTS)
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<80*1024**2:paths.append(path)
    for name in ['cpu_regression_v82.log','cpu_regression_v83.log','cpu_regression_v84.log','cpu_regression_v85.log',
            'cpu_regression_v86.log','dual_side_headroom_report_v0.log','idle_final_summary_v0.log',
            'transport_POLICY_C44_analysis_v0.log']:
        paths.append(ROOT/'logs'/name)
    for path in (ROOT/'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith(TAGS):paths.append(path)
    paths.extend(REPO/record['compressed_path'] for record in compressed)
    paths=sorted(set(paths));assert all(path.is_file() and path.stat().st_size<90*1024**2 for path in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v12.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        bytes=sum(path.stat().st_size for path in paths),closed_GPU_leases=len(closed),
        batch_counter='LIVE',final_resource_closeout=False,goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,'WIP MLLM EPD: imported-KV cost diagnostics, Native EC policy causality and idle synthesis',
        'publication_commit_checkpoint_v12.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V12_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
