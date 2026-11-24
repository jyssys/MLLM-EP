"""Natural-output concurrent token correctness with conservative D admission."""
import asyncio
import json
from pathlib import Path
import random
import time

from . import token_correctness_control as single
from .batch_token_correctness_control import validate
from .client import execute
from .common import ROOT, digest, experiment_deadline, preserve, write
from .frontend_supply_control import retire
from .ipc_ec_control import ownership_empty, stop_proxy
from .llm_probe_control import rpc
from .numerical import embedding_metrics
from .quality_audit import score_request, scorers
from .source_snapshot import snapshot
from .token_split import select_prefix

TAG = 'token_NATURAL_C33'
INDICES = [0, 1, 4, 6, 12, 17, 20, 31]


def compare_full(run, rows, results, strict,fraction=.2,reference_runs=None):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    record = json.loads((ROOT / 'raw/batch_prefix' / f'{run}_D/manifest.json').read_text())
    pool = json.loads((ROOT / 'data/calibration_manifest_template_v2.json').read_text())['requests']
    indices = {pool[index]['request_id']: index for index in INDICES}
    by_id = {row['request_id']: row for row in results}
    scoring = scorers()
    comparisons = []
    for ordinal, row in enumerate(rows):
        base = f'{run}_{ordinal:04d}'
        actual_file = record['full_prompt_tensor_files'][base]
        assert digest(actual_file['path']) == actual_file['SHA256']
        actual = load_file(actual_file['path'])
        index = indices[row['request_id']]
        teacher_run = (reference_runs[row['request_id']] if reference_runs is not None else
            f'token_CORRECTNESS_C30_N8_{index}_TEACHER')
        reference_path = ROOT / 'raw/llm_reference' / f'{teacher_run}_P/manifest.json'
        reference_meta = json.loads(reference_path.read_text())
        assert digest(reference_meta['tensor_file']) == reference_meta['tensor_SHA256']
        reference = load_file(reference_meta['tensor_file'])
        prefix = select_prefix(row['total_input_tokens'], fraction)
        layers = []
        for layer in range(28):
            key = f'prompt_KV_{layer}'
            assert actual[key].shape == reference[key].shape
            segments = {}
            for label, selection in [('prefix', slice(0,prefix)), ('suffix', slice(prefix,None))]:
                metric = embedding_metrics(actual[key][selection], reference[key][selection])
                segments[label] = dict(metric, tolerance_pass=metric['finite'] and
                    metric['relative_L2'] <= .02 and metric['cosine'] >= .999)
            layers.append(dict(layer=layer, strict=torch.equal(actual[key], reference[key]), **segments))
        result = by_id[row['request_id']]
        teacher_path = ROOT / 'raw' / f'{teacher_run}_requests.jsonl'
        teacher = json.loads(teacher_path.read_text().splitlines()[0])
        strict_tokens = result['token_ids'] == teacher['token_ids']
        actual_score = score_request(row,result['text'],scoring)
        reference_score = score_request(row,teacher['text'],scoring)
        passed = all(segment['tolerance_pass'] for layer in layers for segment in [layer['prefix'],layer['suffix']])
        if strict:
            passed = passed and all(r['strict'] for r in layers) and strict_tokens
        comparison = dict(request_id=row['request_id'], base=base, calibration_index=index,
            evidence='CORRECTNESS_ONLY', strict_required=strict, layers=layers,
            strict_tokens=strict_tokens, output_tokens=result['output_tokens'],
            teacher_output_tokens=teacher['output_tokens'], actual_task_score=actual_score,
            teacher_task_score=reference_score, passed=passed,
            actual_file_SHA256=actual_file['SHA256'], reference_manifest_SHA256=digest(reference_path),
            teacher_raw_SHA256=digest(teacher_path),
            scope='Same common compiled numerical contract. Different P batch geometry is allowed only in N8. '
                'Prefix and suffix numerical errors are checked separately. Full official task quality remains pending.')
        comparisons.append(comparison)
        write(ROOT / 'analysis' / f'{run}_full_prompt_comparisons.json', dict(time=time.time(),
            comparisons=comparisons, complete=len(comparisons)==len(rows), task_quality='PENDING'))
        assert passed, 'Stop before performance if full prompt or strict N1 equivalence fails'
    return comparisons


async def cohort(run, rows, url,fraction=.2,reference_runs=None,proxy_name=None):
    contracts = {f'{run}_{i:04d}': dict(prefix_tokens=select_prefix(row['total_input_tokens'],fraction),
        full_prompt_tokens=row['total_input_tokens']) for i,row in enumerate(rows)}
    opened = []
    try:
        for port,role in [(18841,'P'),(18842,'D')]:
            rpc(port,'batch_prefix/begin',dict(name=run+'_'+role,contracts=contracts,
                allow_decode=role=='D',capture_full_prompt=role=='D'))
            opened.append((port,role))
        results = await execute(rows,[url],run,2.0,2026112402,None,
            metadata=dict(evidence='CORRECTNESS_ONLY', manifest_SHA256=digest(ROOT/'data/calibration_manifest_template_v2.json'),
                output='Natural EOS cap2048', admission='Extra suffix starts with zero D HTTP streams.',
                observer='Per-request CPU state copies invalidate all latency and throughput.'))
        assert len(results)==len(rows) and all(r['status']=='MEASURED_ONLINE' for r in results)
    finally:
        for port,role in reversed(opened):
            write(ROOT/'state'/f'{run}_{role}_probe_close.json',rpc(port,'batch_prefix/end',{}))
    ownership_empty(run,'cuda_ipc')
    prefix = validate(run,contracts,include_decode=True)
    comparisons = compare_full(run,rows,results,strict=len(rows)==1,fraction=fraction,reference_runs=reference_runs)
    proxy_name=proxy_name or TAG+'_proxy'
    proxy_records = [json.loads(line) for line in (ROOT/'raw'/f'proxy_{proxy_name}.jsonl').read_text().splitlines()]
    selected = [r for r in proxy_records if r['runtime_request_id'] in contracts]
    assert len(selected)==len(rows)
    for item in selected:
        assert item['token_enabled'] and item['D_admission']['extra_prefill']
        assert item['D_admission']['before']['active_http_requests']==0
        assert item['D_first_generated_token']>=item['D_first_SSE_bytes']
    result = dict(time=time.time(), evidence='CORRECTNESS_ONLY', run=run,
        passed=prefix['passed'] and all(r['passed'] for r in comparisons),
        prefix=prefix, full_prompt_comparisons=comparisons, task_quality='PENDING',
        active_decode_extra_prefill='NOT_RUN',
        scope='Concurrent E/P stages and natural full D decode. Extra D prefill is admitted only after prior D requests finish. '
            'This conservative admission is not a safe-window predictor or a strong-static performance result.')
    write(ROOT/'analysis'/f'{run}_summary.json',result)
    print('NATURAL_TOKEN_STATE_PASS',run,sum(r['strict_tokens'] for r in comparisons),len(rows),flush=True)
    return result


def main():
    preserve()
    assert time.time()<experiment_deadline()
    completion = ROOT/'state/common_native_confirmation_C32_complete.json'
    assert json.loads(completion.read_text())['status']=='COMPLETE'
    for role,gpu in [('E',4),('P',5),('D',6)]:
        lease=json.loads((ROOT/'state'/f'server_ec_COMMON_C32R2_{role}{gpu}.json').read_text())
        assert lease['finished'] and not Path('/proc',str(lease['guardian'])).exists()
    for tag in ['token_CORRECTNESS_C30','token_CORRECTNESS_C30R1']:
        value=json.loads((ROOT/'analysis'/f'{tag}_summary.json').read_text())
        assert value['strict_cases']==value['cases']==15
    assert json.loads((ROOT/'analysis/token_BATCH_C31_summary.json').read_text())['passed']
    manifest=ROOT/'data/calibration_manifest_template_v2.json'
    pool=json.loads(manifest.read_text())['requests']
    rows=[pool[index] for index in INDICES]
    random.Random(2026112404).shuffle(rows)
    plan=ROOT/'configs'/f'{TAG}.json';assert not plan.exists()
    archive=snapshot(TAG)
    write(plan,dict(time=time.time(),status='FROZEN_BEFORE_NATURAL_TOKEN_COHORT',
        evidence='CORRECTNESS_ONLY',source_archive_SHA256=archive['archive_SHA256'],
        manifest_SHA256=digest(manifest),GPU_group=[4,5,6],requests=[r['request_id'] for r in rows],
        sampling='Original BF16 BI0,greedy,Natural EOS cap2048',numerical='Common compiled deterministic-reduction policy',
        safety='Sole D coordinator. A suffix waits for every prior D HTTP stream to finish. '
            'New Native D admission can resume at the first actual generated token, not the SSE role header.',
        query_gate='Every prefill and generated-token MRoPE position matches original metadata.',
        numerical_gate=dict(prefix_and_suffix_relative_L2_max=.02,cosine_min=.999,
            N1='All28 full prompt KV and full output tokens must be strict.',
            N8='Different actual P geometry: record strict output and task scores; full quality remains pending.'),
        no_main_latency_claim=True,no_active_decode_insertion=True,no_Main100_or_Main400_change=True))
    single.TAG=TAG;single.EC=Path('/dev/shm')/f'mllm_epd_20261124_{TAG}'
    names=[];proxy=None;results=[]
    try:
        single.launch_models(names,fixed_reduction=True,deterministic_reductions=True)
        proxy,url=single.launch_proxy(TAG+'_proxy','epd',18849,names,
            suffix_fraction=.2,token_admission='decode_idle')
        results.append(asyncio.run(cohort(TAG+'_N1',[pool[0]],url)))
        results.append(asyncio.run(cohort(TAG+'_N8',rows,url)))
        write(ROOT/'analysis'/f'{TAG}_summary.json',dict(time=time.time(),evidence='CORRECTNESS_ONLY',
            passed=all(r['passed'] for r in results),cohorts=results,task_quality='PENDING',
            main_performance='NOT_RUN',active_decode_extra_prefill='NOT_RUN'))
    except BaseException as error:
        write(ROOT/'state'/f'{TAG}_failure.json',dict(time=time.time(),status='FAILED_EXECUTION',
            error=type(error).__name__+': '+str(error),completed_cohorts=len(results)))
        raise
    finally:
        if proxy is not None:stop_proxy(proxy)
        if names:
            retire([name for name in names if '_E4' not in name])
            retire([name for name in names if '_E4' in name])
    preserve()


if __name__=='__main__':main()
