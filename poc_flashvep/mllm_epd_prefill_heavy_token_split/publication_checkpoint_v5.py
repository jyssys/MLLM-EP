"""Publish closed Native, discovery and rejected correctness evidence without live results."""
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,digest,preserve,write
from .publication_checkpoint_v4 import compress

TAGS=('ec_COMMON_C32','token_NATURAL_C33','token_DISCOVERY_C34','token_FROZEN_C36','vision_BATCH_C38')
DERIVED=TAGS+('common_native_C32','natural_token_C33','token_discovery_C34',
    'frozen_C36','frozen_fraction_C36','encoder_batch_C38','vision_batch_C36_C38',
    'idle_admission_fail_closed','vision_batch_handoff','arrival_contract')
REPORTS=('mllm_epd_common_compiled_native_pilot_v1.md',
    'mllm_epd_common_compiled_native_confirmation_v1.md',
    'mllm_epd_natural_token_correctness_v1.md','mllm_epd_protected_token_discovery_v1.md',
    'mllm_epd_encoder_batch_numerical_root_cause_v1.md')


def main():
    preserve()
    test=ROOT/'logs/cpu_regression_v34.log'
    assert 'Ran 139 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    closed=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not path.name.removeprefix('server_').startswith(TAGS):continue
        row=json.loads(path.read_text())
        assert row['finished'] and not Path('/proc',str(row['guardian'])).exists(),path
        closed.append(path)
    assert closed
    sources={p for p in (ROOT/'raw').iterdir() if p.is_file() and p.suffix in ('.json','.jsonl') and
        p.name.startswith(TAGS+tuple('proxy_'+tag for tag in TAGS))}
    folder=ROOT/'archives/public_checkpoint_v5';manifest=ROOT/'archives/checkpoint_raw_records_v5.json'
    assert not folder.exists() and not manifest.exists()
    folder.mkdir()
    compressed=[compress(p,folder) for p in sorted(sources)]
    write(manifest,dict(time=time.time(),files=compressed,passed=True,
        scope='Closed original responses only. Lossless gzip round-trip verified. '
            'CPU/GPU tensor payloads remain local with hashes. No active timing or shared append-only role log.'))
    readme=ROOT/'README_checkpoint_v5.md';assert not readme.exists()
    readme.write_text('\n'.join(['# MLLM EPD checkpoint v5','',
        'Status: ACTIVE and INCOMPLETE. Main100 and its parent Main400 remain frozen.',
        'Common compiled Native completed3 independent initializations. Natural and equal-output replay remain separate.',
        'Natural TTFT p95 varied across starts. No unfavorable start was removed.',
        'Discovery50 tested all5 predefined static fractions and a frozen idle-fallback candidate.',
        'All static candidates failed the discovery point gate because TTFT p50 regressed.',
        'Idle-fallback10% had lower replay TTFT p95 in one discovery start. This is not confirmed serving gain.',
        'The inherited discovery/heldout task imbalance is disclosed. No frozen split was repaired after performance.',
        'Frozen10% passed N1 and concurrent N8 state checks. Frozen25% failed N8 KV tolerance despite identical outputs.',
        'The25% failure remains rejected. It is not promoted using its discovery goodput.',
        'A diagnostic original-encoder batch change from2 to5 images reproduced embedding and downstream KV differences.',
        'Target processor pixels, tokens, MRoPE and LLM query chunks remained unchanged.',
        'Both same-geometry repeats were strict. The exact unrecorded encoder geometry of the earlier failure remains unknown.',
        'The diagnostic harness originally failed on independent renderer cache identifiers. That failure is preserved.',
        'The correction binds original E exports in a diagnostic-only metadata path. It does not change Main serving.',
        'All these GPU tensor captures are correctness-only. They do not establish kernel cost or online speedup.',
        'Full task quality, prefill-heavy E/P/D simultaneous safe headroom and candidate N3 serving gains remain unproved.',
        'CPU regression:139PASS. The actual diagnostic handoff red/green logs are preserved.',
        'Frozen N3 confirmation code is included. No future confirmation result is claimed.',
        'User checkout/index are preserved through a private Git index. This is not final resource closeout.','']))
    paths=list(SOURCE.glob('*.py'))+[test,manifest,readme]+closed+[ROOT.parent/name for name in REPORTS]
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and name.startswith(DERIVED) and path.stat().st_size<90*1024**2:paths.append(path)
    for kind in ['llm_reference','batch_prefix','encoder_reference']:
        for path in (ROOT/'raw'/kind).glob('*/manifest.json'):
            if path.parent.name.startswith(TAGS):paths.append(path)
    paths += [p for p in (ROOT/'archives').glob('*.source.zip') if p.name.startswith(TAGS)]
    paths += [REPO/row['compressed_path'] for row in compressed]
    paths=sorted(set(paths))
    assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v5.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        total_bytes=sum(p.stat().st_size for p in paths),closed_server_leases=len(closed),
        candidate_confirmation='NOT_RUN at publication freeze',full_quality='PENDING',final_goal_complete=False))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,
        'WIP MLLM EPD: Native N3, protected token discovery and encoder-batch correctness diagnosis',
        'publication_commit_checkpoint_v5.json')
    assert result['passed'] and result['remote_verified'];preserve()
    print('CHECKPOINT_V5_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
