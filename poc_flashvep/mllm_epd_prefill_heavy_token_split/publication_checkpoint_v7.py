"""Publish closed harness failures and conditional Whole correctness, not live N3."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,preserve,write,digest
from .publication_checkpoint_v4 import compress

TAGS=('interference_PROBE_C40R2','interference_PROBE_C40R3',
    'whole_DEFLECTION_C43R3','whole_DEFLECTION_C43R4_CORRECT')
DERIVED=TAGS+('probe_priming_','probe_encoder_state_HTTP_','whole_proxy_log_lifecycle_',
    'whole_baseline_independent_priority_','whole_C43R4_correctness_report_','suffix_cost_evidence_CPU_')


def main():
    preserve();test=ROOT/'logs/cpu_regression_v68.log'
    assert 'Ran 181 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    closed=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAGS):continue
        row=json.loads(path.read_text())
        assert row['finished'] and not Path('/proc',str(row['guardian'])).exists(),path
        closed.append(path)
    proof=ROOT/'analysis/whole_DEFLECTION_C43R4_whole_correctness_audit_v0.json'
    value=json.loads(proof.read_text());assert value['strict_all28_cases']==value['strict_full_outputs']==9
    raw={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith(TAGS+tuple('proxy_'+tag for tag in TAGS))}
    folder=ROOT/'archives/public_checkpoint_v7';manifest=ROOT/'archives/checkpoint_raw_records_v7.json'
    assert not folder.exists() and not manifest.exists();folder.mkdir()
    compressed=[compress(path,folder) for path in sorted(raw)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed original responses only. Lossless gzip round-trip verified. '
            'Same-E tensor payloads remain local with recorded hashes. No live Whole N3 or future probe result.'))
    readme=ROOT/'README_checkpoint_v7.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v7','',
        'Status: ACTIVE and INCOMPLETE. Main100 and parent Main400 remain frozen.',
        'Whole-request full P versus full D passed9 captured cases, with8 unique request IDs and one independent initialization.',
        'All28 prompt KV layers, prompt IDs, MRoPE, prepared inputs and complete greedy outputs matched.',
        'Each comparison reused the same actual original online E output. No tensor or answer was repaired.',
        'This is conditional correctness, not cross-encoder-geometry task quality or serving gain.',
        'The Whole coordinator now uses a distinct log name. The original native log remains unchanged.',
        'Closed failures include the log collision, live-image global-policy rejection and GET-versus-POST state protocol error.',
        'Each failure has a targeted red/green CPU regression. Native model and lifetime assertions remain active.',
        'Private probe priming now consumes the real inline P export and ACK outside timing instead of changing global policy with live images.',
        'The corrected state reader uses the original GET endpoint. No early release or fabricated ACK is permitted.',
        'Whole N3 serving is live and excluded. The next probe preflight waits for its resource closeout and borrows no correctness proof.',
        'The imported-prefix suffix cost study is queued separately. No future cost or safe quantum is claimed.',
        'A separate ABBA Native inline/on-demand descriptor control is frozen and queued. It does not alter the existing inline Token or NativeStrong.',
        'CPU regression:181PASS. Main100 task quality and prefill-heavy safe E/D headroom remain unresolved.',
        'User checkout/index are preserved through a private Git index. This is not final GPU restoration or goal completion.','']))
    paths=list(SOURCE.glob('*.py'))+closed+[manifest,readme,test,proof,
        ROOT.parent/'mllm_epd_probe_priming_lifetime_repair_v1.md',ROOT.parent/'mllm_epd_whole_deflection_correctness_v1.md']
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<80*1024**2:paths.append(path)
    paths.extend(ROOT/'logs'/f'cpu_regression_v{number}.log' for number in range(61,67))
    paths.append(ROOT/'logs/cpu_regression_v67.log')
    paths.append(ROOT/'logs/transport_policy_C44_CPU_v0.log')
    paths.append(ROOT/'configs/ec_INLINE_CAUSAL_C44.json')
    for tag in ['whole_DEFLECTION_C43R4','interference_PROBE_C40R4','suffix_QUANTA_C41R3']:
        paths.append(ROOT/'configs'/f'{tag}.json')
    for path in (ROOT/'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith('whole_DEFLECTION_C43R4_CORRECT'):paths.append(path)
    paths.extend(p for p in (ROOT/'archives').glob('*.source.zip') if p.name.startswith(TAGS))
    paths.extend(REPO/row['compressed_path'] for row in compressed)
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v7.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        bytes=sum(p.stat().st_size for p in paths),closed_GPU_leases=len(closed),
        Whole_correctness='N1+N8 conditional same-E PASS',Whole_N3='LIVE_RESULTS_EXCLUDED',
        additional_probe='WAITING_FOR_WHOLE_RESOURCE',final_goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,'WIP MLLM EPD: same-E Whole correctness and protected probe priming lifecycle repairs',
        'publication_commit_checkpoint_v7.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V7_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
