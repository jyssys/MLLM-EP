"""Publish closed N3 confirmation and diagnostic profiles while research remains active."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,preserve,write,digest
from .publication_checkpoint_v4 import compress

TAGS=('token_CONFIRM_C37','ec_DOCUMENT_C39','interference_PROBE_C40R1')
DERIVED=TAGS+('token_confirmation_C37','token_root_cause_C37','document_C39','task_quality_C37',
    'decode_probe_teacher_rpc_','proxy_identity_','probe_harness_repair_','stale_metadata_proxy_')
REPORTS=('mllm_epd_protected_token_confirmation_v1.md','mllm_epd_protected_token_critical_path_v1.md',
    'mllm_epd_secondary_document_prefill_profile_v1.md','mllm_epd_n3_task_quality_limits_v1.md',
    'mllm_epd_probe_runtime_repair_v1.md')


def main():
    preserve();test=ROOT/'logs/cpu_regression_v60.log'
    assert 'Ran 170 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    closed=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAGS):continue
        row=json.loads(path.read_text())
        assert row['finished'] and not Path('/proc',str(row['guardian'])).exists(),path
        closed.append(path)
    assert closed
    sources={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith(TAGS+tuple('proxy_'+tag for tag in TAGS))}
    for folder in (ROOT/'raw/profiles').glob('ec_DOCUMENT_C39R1_CPC_*'):
        sources.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.csv'))
    for directory in ['analysis','figures']:
        sources.update(p for p in (ROOT/directory).iterdir() if p.is_file() and p.name.startswith(DERIVED) and p.stat().st_size>=80*1024**2)
    folder=ROOT/'archives/public_checkpoint_v6';manifest=ROOT/'archives/checkpoint_raw_records_v6.json'
    assert not folder.exists() and not manifest.exists();folder.mkdir()
    compressed=[compress(path,folder) for path in sorted(sources)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed raw responses, original document-stratum GPU profiles and large derived CSV. Lossless gzip round-trip verified. '
            'No live C40R2/C41R1/C43R1 results, tensor payloads or shared append-only role logs.'))
    readme=ROOT/'README_checkpoint_v6.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v6','',
        'Status: ACTIVE and INCOMPLETE. Main100 and parent Main400 remain unchanged.',
        'Four serving families completed independent N3 on Main100 and the inherited Heldout50.',
        'Natural generation and equal-output replay remain separate.',
        'Protected static10% did not provide qualified serving gain. Idle-fallback did not establish consistent improvement.',
        'Both NativeStrong and common-compiled Native remained unmodified admission controls.',
        'Matched request host accounting shows that conservative D admission wait can exceed P HTTP enclosure savings.',
        'Host enclosures are not exclusive GPU costs. Metric counterfactuals are not runtime interventions.',
        'All37 pre-existing discovery MMLongBench-Doc requests formed a separately frozen source stratum.',
        'Its QPS1/2/3 observer gates failed. Kernel activity and idle results remain DIAGNOSTIC.',
        'No source stratum replaces Main100 or proves E-nonbottleneck prefill-heavy headroom.',
        'Raw source scores and context-cluster uncertainty remain explicit. Full task quality is HOLD.',
        'No unauthorized external answer-extraction API was used.',
        'The first probe harness failed before candidate timing because an old owned coordinator occupied its port.',
        'Actual launcher red/green tests now require process and endpoint identity. Only verified stale owned proxies were closed.',
        'A separate teacher RPC role-contract defect was also corrected with explicit decode-probe opt-in.',
        'No assertion, lifetime ACK, model computation, frozen dataset or SLO was removed.',
        'New probe, suffix-cost and whole-request study code is included. Their future GPU results are not claimed.',
        'CPU regression:170PASS. Failed tests and failed GPU initialization evidence are preserved.',
        'User checkout/index are preserved through a private Git index. This is not final GPU closeout.','']))
    paths=list(SOURCE.glob('*.py'))+[test,manifest,readme]+closed+[ROOT.parent/name for name in REPORTS]
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<80*1024**2:paths.append(path)
    paths.extend(ROOT/'logs'/name for name in ['cpu_regression_v55.log','cpu_regression_v56.log',
        'cpu_regression_v57.log','cpu_regression_v58.log','cpu_regression_v59.log'])
    for tag in ['suffix_QUANTA_C41','whole_DEFLECTION_C43']:
        paths.extend(path for path in [ROOT/'configs'/f'{tag}.json',ROOT/'state'/f'{tag}_controller.json',
            ROOT/'logs'/f'{tag}_v0.log'] if path.exists())
    paths.append(ROOT/'data/secondary_document37_discovery_v0.json')
    paths.append(ROOT/'data/whole_deflection_fresh_holdout_v0.json')
    paths.extend(p for p in (ROOT/'archives').glob('*.source.zip') if p.name.startswith(TAGS))
    paths.extend(REPO/item['compressed_path'] for item in compressed)
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v6.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        total_bytes=sum(p.stat().st_size for p in paths),closed_server_leases=len(closed),
        excluded_live_jobs=['interference_PROBE_C40R2','suffix_QUANTA_C41R1','whole_DEFLECTION_C43R1'],
        full_quality='HOLD',final_goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,'WIP MLLM EPD: protected-token N3, request paths, diagnostic document profiles and probe harness repair',
        'publication_commit_checkpoint_v6.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V6_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
