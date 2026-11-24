"""Publish closed Whole N3 and metric integrity, excluding live probe work."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,preserve,write,digest
from .publication_checkpoint_v4 import compress

TAG='whole_DEFLECTION_C43R4'
DERIVED=(TAG,'main100_actual_','main100_image_','main100_C37_','main100_result_figures_',
    'campaign_metric_audit_v0','campaign_metric_audit_v1','token_online_C37_')


def main():
    preserve();test=ROOT/'logs/cpu_regression_v70.log'
    assert 'Ran 196 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    assert (ROOT/'state'/f'{TAG}_controller_complete.json').exists()
    proof=ROOT/'analysis'/f'{TAG}_audit_v0.json';audit=json.loads(proof.read_text())
    assert audit['metric_recalculation']=='PASS' and not audit['full_GO'] and len(audit['decisions'])==4
    metric=ROOT/'analysis/campaign_metric_audit_v1.json';verification=json.loads(metric.read_text())
    assert verification['status']=='CLOSED_COHORTS_RECALCULATED_NOT_FINAL_CLOSEOUT'
    closed=[];source_archives=[]
    for path in (ROOT/'state').glob('server_'+TAG+'*.json'):
        if path.name.endswith('.source_snapshot.json'):continue
        row=json.loads(path.read_text())
        assert row['finished'] and not Path('/proc',str(row['guardian'])).exists(),path
        closed.append(path)
        archive=Path(row['source_archive'])
        assert digest(archive)==row['source_archive_SHA256']
        source_archives.append(archive)
    assert len(closed)==30, 'One correctness group and nine independent full serving initializations'
    raw={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith((TAG,'proxy_'+TAG))}
    folder=ROOT/'archives/public_checkpoint_v8';manifest=ROOT/'archives/checkpoint_raw_records_v8.json'
    assert not folder.exists() and not manifest.exists();folder.mkdir()
    compressed=[compress(path,folder) for path in sorted(raw)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed Whole original responses, lossless gzip and round-trip SHA256. '
            'Captured tensor payloads remain local with recorded hashes. Live probe, suffix and Native ABBA results are excluded.'))
    readme=ROOT/'README_checkpoint_v8.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v8','',
        'Status: ACTIVE and INCOMPLETE. Main100 and parent Main400 remain frozen.',
        'Whole-request Deflection completed independent N3 for Main100 and fresh-context Holdout50, with Natural and equal-output contracts separated.',
        'NativeStrong keeps the original inline transport and stock compiler. NativeOnDemand keeps the candidate transport but never deflects.',
        'The candidate must pass both controls. No runtime comparison is promoted to full quality or generic Token Split success.',
        'Closed conditional correctness uses the same actual E output for full P versus full D, with all28 KV layers and full greedy outputs checked.',
        'The new one-factor Native transport analysis averages two ABBA periods per policy within each independent start.',
        'It cannot repair absent export RPC in the already inline Static Token or NativeStrong.',
        'A separate metric audit reconstructs TTFT, ITL, TPOT, eight joint-failure categories and goodput from original token timestamps.',
        'Absolute decode protection uses per-request All-ITL AND TPOT, not two separate pass rates.',
        'Six new Main100 figures preserve actual input, output, image/page counts, token-weighted SLO failures and diagnostic P HTTP association.',
        'Qualified low-load physical E/D idle remains distinct from unconfirmed high-load safe headroom.',
        'The owned interference preflight/N3, imported-KV suffix costs and Native ABBA control remain separate live or queued work.',
        'CPU regression:196PASS. Full task quality is pending. No final GPU restoration or goal completion is claimed.',
        'The private publication index preserves the user checkout and staged index.','']))
    paths=list(SOURCE.glob('*.py'))+closed+source_archives+[manifest,readme,test,proof,metric,
        ROOT.parent/'mllm_epd_whole_deflection_online_serving_v1.md',
        ROOT.parent/'mllm_epd_prefill_bottleneck_and_decode_idle.md',
        ROOT.parent/'mllm_epd_dual_side_parallelism_feasibility_v1.md',
        ROOT.parent/'mllm_epd_generic_token_split_online_serving.md']
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<80*1024**2:paths.append(path)
    for log in ['cpu_regression_v69.log','transport_and_whole_analysis_CPU_v0.log',
            'campaign_metric_audit_CPU_v0.log','main100_result_figures_CPU_v0.log']:
        paths.append(ROOT/'logs'/log)
    for path in (ROOT/'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith(TAG):paths.append(path)
    paths.extend(p for p in (ROOT/'archives').glob(TAG+'*.source.zip'))
    paths.extend(REPO/record['compressed_path'] for record in compressed)
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v8.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        bytes=sum(p.stat().st_size for p in paths),closed_GPU_leases=len(closed),
        Whole_N3='CLOSED_AND_RECALCULATED',task_quality='PENDING',
        live_probe_suffix_and_ABBA_results='EXCLUDED',final_goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,'WIP MLLM EPD: Whole-request N3, raw metric verification and Main100 failure figures',
        'publication_commit_checkpoint_v8.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V8_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
