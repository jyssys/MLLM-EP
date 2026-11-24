"""Publish closed correctness and observer evidence without live-run records."""
import gzip
import hashlib
import json
from pathlib import Path
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,digest,preserve,write


TAGS=('token_CORRECTNESS_C15R2','token_NUMERICAL_C17','token_NUMERICAL_C18',
      'ec_GPU_ENCLOSURES_C19','ec_GPU_ENCLOSURES_C20','token_NUMERICAL_C21',
      'token_NUMERICAL_C24','prefix_QUANTA_C22')


def main():
    preserve();test=ROOT/'logs/cpu_regression_v17.log'
    assert 'Ran 99 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    closed=[]
    for path in (ROOT/'state').glob('server_*.json'):
        if path.name.endswith('.source_snapshot.json') or not any(path.name.startswith('server_'+tag) for tag in TAGS):continue
        value=json.loads(path.read_text())
        assert value['finished']
        assert not Path('/proc',str(value['guardian'])).exists(),path
        closed.append(path)
    assert closed
    folder=ROOT/'archives/public_checkpoint_v3';folder.mkdir(exist_ok=True)
    manifest=ROOT/'archives/checkpoint_raw_records_v3.json';assert not manifest.exists()
    compressed=[]
    for source in sorted((ROOT/'raw').glob('*.jsonl')):
        if not source.name.startswith(TAGS+tuple('proxy_'+tag for tag in TAGS)):continue
        original=digest(source);target=folder/(source.name+'.gz')
        with source.open('rb') as src,target.open('xb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0,compresslevel=6) as archive:
                while block:=src.read(8*1024**2):archive.write(block)
        hasher=hashlib.sha256();size=0
        with gzip.open(target,'rb') as stream:
            while block:=stream.read(8*1024**2):hasher.update(block);size+=len(block)
        assert hasher.hexdigest()==original==digest(source) and size==source.stat().st_size
        compressed.append(dict(source=str(source.relative_to(REPO)),source_SHA256=original,source_bytes=size,
            compressed_path=str(target.relative_to(REPO)),compressed_SHA256=digest(target),
            compressed_bytes=target.stat().st_size,lossless_verification='PASS'))
    write(manifest,dict(time=time.time(),passed=True,files=compressed,
        scope='Closed correctness/diagnostic request records. Live C25/C26 and shared role logs are excluded. '
              'Tensor payloads and full kernel traces remain local with their original hashes. Not final closeout.'))
    readme=ROOT/'README_checkpoint_v3.md';assert not readme.exists()
    readme.write_text('\n'.join([
        '# MLLM EPD checkpoint v3','',
        'Status: ACTIVE and INCOMPLETE. Main100 remains active. Main400 and earlier evidence remain unchanged.',
        'Original eager P/D token continuation passed15/15 isolated correctness cases, including splits inside image spans.',
        'This is not compiled C3 token serving or a full task-quality pass.',
        'Stock compiled execution differed first at the first post-attention RMSNorm.',
        'Actual P/D reduction blocks were2048/4096, despite matching original inputs and generated kernel source.',
        'A common compiled-only dynamic_scale_rblock=False control made that norm and four following GEMM captures strict.',
        'Full prompt KV strict equality increased only from1/28 to2/28 layers. The full compiled gate remains unmet.',
        'All16 greedy tokens matched in this one diagnostic request. No performance conclusion follows.',
        'QPS0.5 CUDA-event clean/profile/clean passed the frozen5% observer gate in one independent initialization.',
        'The events bracket execute calls. They do not prove exclusive kernel activity, physical idle or safe headroom.',
        'QPS2/4 observer failures remain diagnostic. Low-load qualification does not qualify overloaded runs.',
        'An extra original full model co-resided with E. Memory fit is separate from safe extra-prefill execution.',
        'E and D helpers each passed all28 KV layers for128/256/512/1024/2048-token real prefixes.',
        'The helpers used original online E embeddings. These were correctness-only, not suffix or task-completion results.',
        'The first prefix probe failed because an unused image received an inline descriptor with no clone ACK.',
        'The corrected probe uses existing on-demand descriptors. Issued descriptors still require clone completion before release.',
        'The preserved CPU red test, CPU green tests and10/10 GPU cases verify this ownership correction.',
        'Full task quality, compiled token C3, additional-prefill safety and serving improvement remain unresolved.',
        'Live C25 low-load CUPTI and C26 prefix-cost records are excluded from this checkpoint.',
        'CPU regression before the final cost-followup edits:99PASS. Later prefix tests:5PASS.',
        'No complete goal or final GPU closeout is claimed. Approved burners run on unused GPUs.','']))
    paths=list(SOURCE.glob('*.py'))+[test,manifest,readme]
    for name in ['mllm_epd_generic_token_split_correctness.md','mllm_epd_cuda_event_observer_diagnostic_v1.md',
        'mllm_epd_low_load_event_bracket_validation_v1.md','mllm_epd_compiled_numerical_root_cause_v1.md',
        'mllm_epd_encoder_decoder_prefix_quantum_correctness_v1.md','mllm_epd_docvqa_raw_answer_score_floor_v1.md']:
        paths.append(ROOT.parent/name)
    prefixes=TAGS+('compiled_numerical_C21_C24','token_C15R2','token_C17_C18',
                  'token_correctness_checkpoint','DocVQA_raw_answer_floor','low_load_C20')
    for directory in ['analysis','configs','state','logs','figures']:
        for path in (ROOT/directory).iterdir():
            name=path.name.removeprefix('server_')
            if path.is_file() and (name.startswith(prefixes) or
                (directory=='logs' and name.startswith(('prefix_probe_','gemm_','norm_probe_','fixed_reduction_')))):
                if path.stat().st_size<90*1024**2:paths.append(path)
    for path in (ROOT/'raw/llm_reference').glob('*/manifest.json'):
        if path.parent.name.startswith(TAGS):paths.append(path)
    for path in (ROOT/'raw/gpu_enclosures').glob('*/*/*'):
        if path.is_file() and path.parent.parent.name.startswith(TAGS):paths.append(path)
    paths.extend(REPO/r['compressed_path'] for r in compressed)
    paths.extend((ROOT/'archives').glob('*.source.zip'))
    paths.append(ROOT/'archives/token_C17_compiled_artifacts_v0.tar.gz')
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v3.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        total_bytes=sum(p.stat().st_size for p in paths),closed_server_leases=len(closed),
        excluded_live_jobs=['C25','C26'],final_goal_complete=False,
        user_checkout_and_staged_index='Preserved through a private Git index'))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE;helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,
        'WIP MLLM EPD: exact eager token checks, compiled norm cause, event qualification and prefix ownership fix',
        'publication_commit_checkpoint_v3.json')
    assert result['passed'] and result['remote_verified']
    preserve();print('CHECKPOINT_V3_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
