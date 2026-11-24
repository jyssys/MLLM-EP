"""Publish closed baseline and memory evidence through the private index."""
import gzip
import hashlib
import json
import time

from .common import ROOT,REPO,SOURCE,BASE,BRANCH,digest,preserve,write


def main():
    preserve()
    test=ROOT/'logs/cpu_regression_token_v14.log'
    assert 'Ran 85 tests' in test.read_text() and test.read_text().rstrip().endswith('OK')
    assert json.loads((ROOT/'state/main100_baseline_confirmation_C14_complete.json').read_text())['status']=='COMPLETE'
    baseline=ROOT/'analysis/main100_baseline_confirmation_C14_v0.json'
    assert json.loads(baseline.read_text())['status']=='RUNTIME_MEASURED_QUALITY_PENDING'
    memory=ROOT/'analysis/encoder_LLM_MEMORY_C16.json'
    assert json.loads(memory.read_text())['status']=='OBSERVED_CO_RESIDENCY_FIT'
    folder=ROOT/'archives/public_checkpoint_v2';folder.mkdir(exist_ok=True)
    manifest=ROOT/'archives/checkpoint_raw_records_v2.json'
    assert not manifest.exists()
    compressed=[]
    for source in sorted((ROOT/'raw').glob('*.jsonl')):
        if not source.name.startswith(('base_C14_','proxy_base_C14_',
                'token_CORRECTNESS_C15R1_','proxy_token_CORRECTNESS_C15R1_',
                'encoder_LLM_MEMORY_C16_')):continue
        original=digest(source);target=folder/(source.name+'.gz')
        with source.open('rb') as src,target.open('xb') as dst:
            with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0,compresslevel=6) as archive:
                while block:=src.read(8*1024**2):archive.write(block)
        hasher=hashlib.sha256();size=0
        with gzip.open(target,'rb') as stream:
            while block:=stream.read(8*1024**2):hasher.update(block);size+=len(block)
        assert hasher.hexdigest()==original==digest(source) and size==source.stat().st_size
        compressed.append(dict(source=str(source.relative_to(REPO)),source_SHA256=original,
            source_bytes=size,compressed_path=str(target.relative_to(REPO)),
            compressed_SHA256=digest(target),compressed_bytes=target.stat().st_size,
            lossless_verification='PASS'))
    write(manifest,dict(time=time.time(),passed=True,files=compressed,
        scope='Closed C14 baseline and C15R1 request/proxy records only. Live C17R1 and shared role logs are excluded. '
            'Nested tensors, profiles and media remain local. This is not the final evidence archive.'))
    from .source_snapshot import snapshot
    source=snapshot('publication_checkpoint_v2')
    readme=ROOT/'README_checkpoint_v2.md'
    assert not readme.exists()
    readme.write_text('\n'.join([
        '# MLLM EPD checkpoint v2','',
        'Status: ACTIVE and INCOMPLETE. Earlier reports and checkpoints remain unchanged.',
        'Main100 is active. Main400 remains available without modification.',
        'Agg DP3, E+PD DP2 and Native EPD completed the common three-GPU Natural baseline sweep.',
        'QPS2 and QPS4 use three independent server starts. Other QPS points use one discovery start.',
        'The shared QPS grid is0.5,1,2,3,4,6. Actual points and start-level intervals have corresponding CSV files.',
        'Native QPS2 reuses the qualified C11 runtime records. It does not repeat them under a new label.',
        'Full task quality remains PENDING. Runtime timing is not a task-quality pass.',
        'The official Mantis scorer retains its48% result for E+PD S2, versus52% for the reference.',
        'A diagnostic shows that the single different score concerns two responses with the same final option A.',
        'The official parser does not recognize one response format. No answer or official score was changed.',
        'C15 failed before GPU loading because its preflight counted source receipts as GPU ownership leases.',
        'The preserved red test and corrected lease check distinguish source receipts from actual server leases.',
        'C15R1 ran the first exact token boundary inside an image span.',
        'All28 imported prefix KV layers were byte-exact. Prompt, MRoPE and prepared embeddings matched.',
        'All16 output tokens matched the chunk-aligned teacher. Full suffix KV strict equality failed.',
        'No token performance run followed this failed gate. Separate numerical diagnosis continues.',
        'C16 loaded an extra original full model beside the encoder, with about8GiB of KV storage.',
        'Three predefined real input stresses completed. All68 encoder features matched before and after loading.',
        'This proves observed memory fit only. Extra prefill safety and maximum encoder batch capacity remain unmeasured.',
        'CUPTI observer failures remain diagnostic. No decode-safe headroom or token-split speedup is claimed.',
        'Approved burners remain on unused GPUs. This checkpoint is not final resource closeout.','',
        'Main100 SHA256:910506f24984e8f83e433c169b0a03d06c89396e1ef8a5f66b514d97406844d1.',
        'Model/processor revision:cc594898137f460bfe9f0759e9844b3ce807cfb5.',
        'CPU regression result:85PASS. GPU correctness and task-quality gates remain separate.',
        'Closed raw subset:archives/checkpoint_raw_records_v2.json.',
        'The final campaign must verify the remaining gated scope, archive evidence and restore all approved burners.','']))
    paths=list(SOURCE.glob('*.py'))+[test,baseline,memory,manifest,readme]
    for name in ['mllm_epd_main100_baseline_motivation_v1.md','mllm_epd_mantis_parser_diagnostic_v1.md',
                 'mllm_epd_encoder_llm_memory_diagnostic_v1.md']:paths.append(ROOT.parent/name)
    for directory,patterns in [
        ('analysis',['base_C14_*.json','C14_*.json','main100_baseline_confirmation_C14_*.csv',
            'encoder_LLM_memory_C16*.json','token_CORRECTNESS_C15R1*.json']),
        ('configs',['base_C14_*.json','main100_baseline_confirmation_C14*.json',
            'token_CORRECTNESS_C15*.json','encoder_LLM_MEMORY_C16*.json']),
        ('state',['server_base_C14_*.json','server_token_CORRECTNESS_C15R1*.json',
            'server_encoder_LLM_MEMORY_C16*.json','main100_baseline_confirmation_C14_complete.json',
            'token_CORRECTNESS_C15R1_failure.json']),
        ('figures',['main100_C14_*.png','main100_C14_*.pdf','main100_C14_*.csv',
            'encoder_LLM_memory_C16*.png','encoder_LLM_memory_C16*.pdf','encoder_LLM_memory_C16*.csv']),
        ('logs',['token_C15_preflight_*.log','attention_RPC_*.log'])]:
        for pattern in patterns:paths.extend((ROOT/directory).glob(pattern))
    paths.extend(REPO/r['compressed_path'] for r in compressed)
    paths.extend((ROOT/'archives').glob('*.source.zip'))
    paths=sorted(set(paths));assert all(p.is_file() and p.stat().st_size<90*1024**2 for p in paths)
    selection=ROOT/'state/checkpoint_publication_selection_v2.json'
    write(selection,dict(time=time.time(),status='INCOMPLETE_RESEARCH_CHECKPOINT',files=len(paths),
        source_archive_SHA256=source['archive_SHA256'],total_bytes=sum(p.stat().st_size for p in paths),
        excluded_live_job='C17R1 numerical correctness diagnostic',final_goal_complete=False,
        user_checkout_and_staged_index='Preserved through a private Git index'))
    paths.append(selection)
    from ..dense_ar_p_critical_path import publication as helper
    helper.ROOT,helper.REPO,helper.SOURCE=ROOT,REPO,SOURCE
    helper.BASE,helper.BRANCH=BASE,BRANCH
    helper.preserve,helper.write,helper.digest=preserve,write,digest
    result=helper.commit_paths(paths,
        'WIP MLLM EPD Main100: common N3 baselines, preserved token strict failure and encoder model memory fit',
        'publication_commit_checkpoint_v2.json')
    assert result['passed'] and result['remote_verified']
    preserve();print('CHECKPOINT_V2_REMOTE_VERIFIED',result['remote_SHA'],flush=True)


if __name__=='__main__':main()
