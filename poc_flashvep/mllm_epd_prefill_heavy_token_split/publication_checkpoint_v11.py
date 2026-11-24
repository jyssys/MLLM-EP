"""Publish completed N3 probe failures without a safe-headroom claim."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,preserve,write,digest
from .publication_checkpoint_v4 import compress

TAG='interference_PROBE_C40R4_B2'
DERIVED=('interference_PROBE_C40R4','probe_failure_tail_C40','probe_C40_',
    'delivery_requirement_inventory_v2','delivery_requirement_figures_v2')


def main():
    preserve();test=ROOT/'logs/cpu_regression_v81.log'
    assert 'Ran 223 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    completed=ROOT/'state/interference_PROBE_C40R4_controller_complete.json';assert completed.exists()
    summary=ROOT/'analysis/interference_PROBE_C40R4_summary.json';data=json.loads(summary.read_text())
    assert data['status']=='COMPLETE' and len(data['starts'])==3
    gate=ROOT/'analysis/interference_PROBE_C40R4_audit_v0.json';checks=json.loads(gate.read_text())
    assert len(checks['decisions'])==8 and all(not r['observed_HTTP_protection_gate'] for r in checks['decisions'])
    tail=ROOT/'analysis/probe_failure_tail_C40_audit_v0.json';tail_check=json.loads(tail.read_text())
    assert tail_check['actual_probe_cohorts']==24 and tail_check['p99_below_threshold_but_request_decode_gate_failed']==4
    inventory=ROOT/'analysis/delivery_requirement_inventory_v2.json';items=json.loads(inventory.read_text())
    assert not items['goal_complete'] and sum(r['artifact_available'] for r in items['figures'])>=24
    metric=ROOT/'analysis/campaign_metric_audit_v3.json';audit=json.loads(metric.read_text())
    assert audit['cohorts']>=171 and audit['requests']>=15000
    closed=[];archives=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAG):continue
        row=json.loads(path.read_text());assert row['finished'] and not Path('/proc',str(row['guardian'])).exists()
        closed.append(path);archive=Path(row['source_archive'])
        assert digest(archive)==row['source_archive_SHA256'];archives.append(archive)
    assert len(closed)==7
    raw={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith((TAG,'proxy_'+TAG,'NVML_'+TAG))}
    folder=ROOT/'archives/public_checkpoint_v11';manifest=ROOT/'archives/checkpoint_raw_records_v11.json'
    assert not folder.exists() and not manifest.exists();folder.mkdir()
    compressed=[compress(path,folder) for path in sorted(raw)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed final probe block. Earlier closed blocks are in v9/v10 archives. '
            'Lossless gzip round-trip checks. Live suffix/Native-policy/counter results and shared append-only logs are excluded.'))
    readme=ROOT/'README_checkpoint_v11.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v11','',
        'Status: ACTIVE and INCOMPLETE. This checkpoint publishes completed N3 interference evidence.',
        'All three prospective probe blocks completed on the same E/P/D GPU group.',
        'Every E128/E512/D128/D512 Natural and Replay condition failed the complete frozen protection gate.',
        'Some metrics improved. A point improvement does not erase other failed gates or wide confidence intervals.',
        'All conditions executed additional real prefix work. Their discarded samples do not count toward client task throughput.',
        'Four of24 probe cohorts had client ITL p99 below34ms but failed the request-level95% decode gate.',
        'That gate requires every ITL and request TPOT to pass. A pooled p99 does not replace it.',
        'All request/token failure combinations remain in the CSV records. No failed start was removed.',
        'Worker-level GPU competition versus host or client delivery was not measured for these clean cohorts.',
        'Fixed period order and numerical/batch variation limit a unique runtime-cause attribution.',
        'No encode-safe or decode-safe additional-prefill success is claimed.',
        'E/helper allocator and shared device memory snapshots are published separately from logical weight and KV bytes.',
        'The helper used8GiB KV. Native D used42GiB. This is not equal KV capacity or maximum encoder-batch validation.',
        'Imported-KV suffix correctness and cost belong to a separate study, not these no-import prefix probes.',
        'At this checkpoint the suffix, Native transport policy and actual batch studies remain live or queued.',
        'CPU regression:223PASS. Full task quality and high-load prefill-heavy feasibility remain unresolved.',
        'Final reports, final resource closeout and final campaign publication remain pending.','']))
    paths=list(SOURCE.glob('*.py'))+closed+archives+[test,completed,summary,gate,tail,inventory,metric,manifest,readme,
        ROOT.parent/'mllm_epd_encoder_decoder_interference_probe_v1.md',
        ROOT.parent/'mllm_epd_probe_decode_tail_and_failure_attribution_v1.md']
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<80*1024**2:paths.append(path)
    for name in ['campaign_metric_audit_v3.json','campaign_metric_audit_v3.csv']:
        paths.append(ROOT/'analysis'/name)
    for name in ['cpu_regression_v80.log','probe_failure_diagnostic_v0.log','campaign_metric_audit_v3.log']:
        paths.append(ROOT/'logs'/name)
    for path in (ROOT/'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith(TAG):paths.append(path)
    paths.extend(REPO/record['compressed_path'] for record in compressed)
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v11.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        bytes=sum(p.stat().st_size for p in paths),closed_GPU_leases=len(closed),
        N3_probe='COMPLETE_PROTECTION_NOT_CONFIRMED',goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,'WIP MLLM EPD: N3 E/D prefix probe protection failures and request-level tail audit',
        'publication_commit_checkpoint_v11.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V11_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
