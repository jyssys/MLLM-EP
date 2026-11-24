"""Publish closed second probe block and scoped figures, not unfinished N3."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,preserve,write,digest
from .publication_checkpoint_v4 import compress

TAG='interference_PROBE_C40R4_B1'
DERIVED=(TAG,'idle_scope_','parallelism_feasibility_','delivery_requirement_')


def main():
    preserve();test=ROOT/'logs/cpu_regression_v79.log'
    assert 'Ran 220 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    first=ROOT/'analysis'/f'{TAG}_summary.json';assert json.loads(first.read_text())['status']=='COMPLETE'
    inventory=ROOT/'analysis/delivery_requirement_inventory_v1.json'
    check=json.loads(inventory.read_text());assert not check['goal_complete']
    assert sum(r['artifact_available'] for r in check['figures'])==22
    metric=ROOT/'analysis/campaign_metric_audit_v2.json';audit=json.loads(metric.read_text())
    assert audit['cohorts']>131 and audit['requests']>11000
    closed=[];source_archives=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAG):continue
        row=json.loads(path.read_text());assert row['finished'] and not Path('/proc',str(row['guardian'])).exists()
        closed.append(path);archive=Path(row['source_archive'])
        assert digest(archive)==row['source_archive_SHA256'];source_archives.append(archive)
    assert len(closed)==7
    raw={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith((TAG,'proxy_'+TAG,'NVML_'+TAG))}
    folder=ROOT/'archives/public_checkpoint_v10';manifest=ROOT/'archives/checkpoint_raw_records_v10.json'
    assert not folder.exists() and not manifest.exists();folder.mkdir()
    compressed=[compress(path,folder) for path in sorted(raw)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed B1 probe and original client responses. Lossless gzip with round-trip SHA256. '
            'Live B2, suffix, Native ABBA, counter and shared append-only logs are excluded. No N3 verdict.'))
    readme=ROOT/'README_checkpoint_v10.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v10','',
        'Status: ACTIVE and INCOMPLETE. Final reports and resource closeout remain pending.',
        'The second independent probe block completed. This checkpoint publishes closed B1 evidence only.',
        'The third block remains live. No N3 protection verdict is inferred from two blocks.',
        'The actual probe is unrelated prefix work. Its discarded sample does not count toward client throughput.',
        'The memory analyzer separates shared GPU4 device use from each process allocator.',
        'E and helper device use is not added twice. Reserved memory is not model-weight memory.',
        'Logical parameters and unique KV storage remain separate. Peaks cover the worker lifetime, not each cohort.',
        'Actual low-load GPU activity and queue figures now include corresponding CSV and PDF files.',
        'C25 low load passed the observer gate. C27 and C11 high-load observations remain DIAGNOSTIC.',
        'C25 and C27 share the explicitly audited core/flag fingerprint, but their observation windows differ.',
        'C11 has a different fingerprint. The figures do not join these points into one runtime curve.',
        'Scheduler queue dots are actual observations. The figures do not interpolate missing states.',
        'The future-parallelism summary is STRUCTURAL_ACCOUNTING, not implemented three-GPU speedup.',
        'The intermediate inventory validates artifact availability and basic file integrity only.',
        'It does not infer correctness, task quality, safe headroom or goal completion from file presence.',
        'Closed cohort metrics were independently recalculated from token emission timestamps and frozen SLOs.',
        'CPU regression:220PASS. Protected checkout, index and previous evidence remain unchanged.',
        'Imported-KV suffix, Native EC policy and actual batch-counter studies remain queued or live.',
        'Full task quality and actual high-load safe headroom remain unresolved.','']))
    paths=list(SOURCE.glob('*.py'))+closed+source_archives+[test,first,inventory,metric,manifest,readme,
        ROOT.parent/'mllm_epd_idle_scope_and_required_figures_v1.md',
        ROOT.parent/'mllm_epd_idle_scope_and_required_figures_v2.md']
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<80*1024**2:paths.append(path)
    for name in ['campaign_metric_audit_v2.csv','campaign_metric_audit_v2.json']:
        paths.append(ROOT/'analysis'/name)
    for name in ['cpu_regression_v75.log','cpu_regression_v76.log','cpu_regression_v77.log',
            'cpu_regression_v78.log','campaign_metric_audit_v2.log']:
        paths.append(ROOT/'logs'/name)
    for path in (ROOT/'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith(TAG):paths.append(path)
    paths.extend(REPO/record['compressed_path'] for record in compressed)
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v10.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        bytes=sum(p.stat().st_size for p in paths),closed_GPU_leases=len(closed),
        actual_N3_probe_verdict='PENDING_NOT_PUBLISHED',goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,'WIP MLLM EPD: closed second probe start, scoped GPU idle figures and delivery audit',
        'publication_commit_checkpoint_v10.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V10_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
