"""Isolated exact token continuation against a chunk-aligned full teacher."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

from .client import one, payload
from .common import ROOT, REPO, PYTHON, digest, preserve, write, experiment_deadline
from .frontend_supply_control import ROLES, wait_ready, retire
from .ipc_ec_control import stop_proxy, ownership_empty
from .llm_probe_control import rpc
from .source_snapshot import snapshot
from .token_split import TEACHER_KEY, select_prefix

TAG = 'token_CORRECTNESS_C15'
EC = Path('/dev/shm') / f'mllm_epd_20261124_{TAG}'


def wait_baselines():
    complete = ROOT / 'state/main100_baseline_confirmation_C14_complete.json'
    while not complete.exists():
        proc = Path('/proc/550710')
        assert proc.exists() and proc.stat().st_uid == os.getuid(), 'Prior baseline became terminal without a completion record'
        assert b'mllm_epd_prefill_heavy_token_split.baseline_confirmation' in (proc / 'cmdline').read_bytes()
        assert time.time() < experiment_deadline(), 'No new GPU experiment after the reserved cutoff'
        time.sleep(1)
    assert json.loads(complete.read_text())['status'] == 'COMPLETE'
    paths = {p for p in (ROOT / 'state').glob('server_base_C14_*.json')
             if not p.name.endswith('.source_snapshot.json')}
    roles = dict(agg=['A4', 'A5', 'A6'], e_pd=['E4', 'P5', 'P6'],
                 native=['E4', 'P5', 'D6'])
    expected = {ROOT / 'state' / f'server_base_C14_{family}_S{start}_{role}.json'
                for family, members in roles.items() for start in [1, 2, 3]
                for role in members}
    assert paths == expected, 'Missing or unexpected baseline GPU ownership lease'
    leases = [json.loads(p.read_text()) for p in paths]
    assert all(r['finished'] and not Path('/proc', str(r['guardian'])).exists() for r in leases)


def launch_models(names,diagnostic_eager=False,fixed_reduction=False,deterministic_reductions=False):
    assert not EC.exists(), 'Do not reuse a CUDA ownership epoch'
    assert not names
    for role, gpu, port, side in ROLES:
        name = f'{TAG}_{role[0].upper()}{gpu}'
        names.append(name)
        args = [str(PYTHON), '-m', 'poc_flashvep.mllm_epd_prefill_heavy_token_split.servers',
            'start', '--name', name, '--gpu', str(gpu), '--role', role, '--port', str(port),
            '--side-port', str(side), '--ec-path', str(EC), '--observe', '--ec-transport', 'cuda_ipc']
        if role == 'encoder':
            args += ['--encoder-native-ops', '--compiled-encoder', '--encoder-sync-scheduling',
                     '--mm-cache-gb', '0', '--renderer-workers', '2']
        else:
            args += ['--token-split']
            if diagnostic_eager:args += ['--eager-reference']
            if fixed_reduction:args += ['--fixed-reduction']
            if deterministic_reductions:args += ['--deterministic-reductions']
        subprocess.run(args, cwd=REPO, check=True)
    wait_ready(names)
    for port in [18841, 18842]:
        rpc(port, 'ec_ipc/ack_policy', dict(mode='file'))
    reply = rpc(18840, 'ec_ipc/inline_policy', dict(enabled=True))
    assert reply[0]['inline_role'] == 'prefill'
    return names


def launch_proxy(label, mode, port, names, prefix=0, decoder_port=None,
                 suffix_fraction=0.,token_admission='none',token_policy='static'):
    assert not (prefix and suffix_fraction)
    decoder_port=decoder_port or (18841 if mode=='e_pd' else 18842)
    command = [str(PYTHON), '-m', 'poc_flashvep.mllm_epd_prefill_heavy_token_split.proxy',
        '--mode', mode, '--name', label, '--encoder', 'http://127.0.0.1:18840',
        '--prefill', 'http://127.0.0.1:18841', '--decoders',
        f'http://127.0.0.1:{decoder_port}',
        '--port', str(port), '--processor-handoff', 'grid_only', '--release-encoder-ipc']
    if prefix:
        command += ['--token-prefix-tokens', str(prefix)]
    if suffix_fraction:
        command += ['--token-suffix-fraction',str(suffix_fraction)]
    if token_admission!='none':
        command += ['--token-admission',token_admission,'--token-policy',token_policy]
    lease = json.loads((ROOT / 'state' / f'server_{names[0]}.json').read_text())
    with (ROOT / 'logs' / f'{label}.log').open('xb') as stream:
        process = subprocess.Popen(command, cwd=lease['executed_source_cwd'], stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT / 'state' / f'{label}.json', dict(pid=process.pid, sid=process.pid, command=command,
        scope='Correctness proxy, not an online performance arm'))
    url = f'http://127.0.0.1:{port}'
    deadline = time.time() + 30
    while True:
        assert process.poll() is None
        try:
            with urllib.request.urlopen(url + '/health', timeout=2) as response:
                if response.status == 200:
                    return process, url
        except OSError:
            pass
        if time.time() >= deadline:
            raise TimeoutError('Inspect the same live correctness proxy before any restart')
        time.sleep(.2)


async def request(row, url, run, teacher_prefix=None):
    import aiohttp
    body = payload(row, run, None)
    if teacher_prefix is not None:
        body['kv_transfer_params'] = {TEACHER_KEY: dict(
            prefix_tokens=teacher_prefix, full_prompt_tokens=row['total_input_tokens'])}
    assert not (ROOT / 'raw' / f'{run}_requests.jsonl').exists()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), trust_env=False) as session:
        value = await one(session, url, row, body, time.monotonic(), run, 0)
    assert value['status'] == 'MEASURED_ONLINE', value.get('error')
    ownership_empty(run, 'cuda_ipc')
    return value


def capture(row, run, url, ports, teacher_prefix=None,include_attention=False,include_gemm=False,attention_layer=0):
    opened = []
    try:
        for port, role in ports:
            rpc(port, 'llm_probe/begin', dict(name=run + '_' + role, max_logits=16,
                include_decode_inputs=True,include_attention=include_attention,include_gemm=include_gemm,attention_layer=attention_layer))
            opened.append((port, role))
        return asyncio.run(request(row, url, run, teacher_prefix))
    finally:
        for port, role in reversed(opened):
            write(ROOT / 'state' / f'{run}_{role}_probe_close.json', rpc(port, 'llm_probe/end', {}))


def prefill_events(record, left, right):
    events = []
    cursor = left
    for event in record['events']:
        start = event['start']
        if start >= right:
            continue
        end = min(right, start + event['query_tokens'])
        assert start == cursor and end > start, ('Gap or duplicated query', cursor, event)
        events.append(event)
        cursor = end
    assert cursor == right
    return events


def compare(row, prefix, teacher_run, token_run, teacher_result, token_result):
    import torch
    from safetensors.torch import load_file
    from .numerical import embedding_metrics
    torch.set_num_threads(2)
    records = {}
    values = {}
    for label, run, role in [('T', teacher_run, 'P'), ('P', token_run, 'P'), ('D', token_run, 'D')]:
        record = json.loads((ROOT / 'raw/llm_reference' / f'{run}_{role}/manifest.json').read_text())
        assert record['snapshot'] is not None and digest(record['tensor_file']) == record['tensor_SHA256']
        records[label] = record
        values[label] = load_file(record['tensor_file'])
    length = row['total_input_tokens']
    teacher_events = prefill_events(records['T'], 0, length)
    p_events = prefill_events(records['P'], 0, prefix)
    d_events = prefill_events(records['D'], prefix, length)
    assert any(e['start'] + e['query_tokens'] == prefix for e in teacher_events), 'Teacher did not honor the exact chunk boundary'
    teacher_by_start = {event['start']: event for event in teacher_events}
    input_checks = []
    for label, events in [('P', p_events), ('D', d_events)]:
        for event in events:
            ref = teacher_by_start[event['start']]
            assert ref['query_tokens'] == event['query_tokens'], 'Teacher and token query geometry differ'
            for key in ['positions', 'inputs_embeds']:
                actual_key = key + '_' + str(event['index'])
                target_key = key + '_' + str(ref['index'])
                assert actual_key in values[label] and target_key in values['T']
                input_checks.append(dict(role=label, start=event['start'], field=key,
                    strict=torch.equal(values[label][actual_key], values['T'][target_key])))
    fields = dict(
        full_prompt_ids=torch.equal(values['T']['prompt_token_ids'], values['D']['prompt_token_ids']),
        prefix_prompt_ids=torch.equal(values['T']['prompt_token_ids'][:prefix], values['P']['prompt_token_ids']),
        full_mrope=torch.equal(values['T']['full_prompt_mrope'], values['D']['full_prompt_mrope']),
        prefix_mrope=torch.equal(values['T']['full_prompt_mrope'][:, :prefix], values['P']['full_prompt_mrope']),
        full_delta=records['T']['snapshot']['mrope_delta'] == records['D']['snapshot']['mrope_delta'])
    layers = []
    for layer in range(28):
        key = f'prompt_KV_{layer}'
        source, destination, reference = values['P'][key], values['D'][key], values['T'][key]
        assert source.shape[0] == prefix and destination.shape[0] == reference.shape[0] == length
        row_metric = dict(layer=layer, imported_prefix_strict=torch.equal(source, destination[:prefix]),
            P_teacher_prefix_strict=torch.equal(source, reference[:prefix]),
            full_prompt_strict=torch.equal(destination, reference))
        if not row_metric['P_teacher_prefix_strict']:
            row_metric['P_teacher_prefix_error'] = embedding_metrics(source, reference[:prefix])
        if not row_metric['full_prompt_strict']:
            row_metric['full_prompt_error'] = embedding_metrics(destination, reference)
        layers.append(row_metric)
    metadata_key = hashlib.sha256((token_run + '_0000').encode()).hexdigest()
    metadata = json.loads((EC / 'processed_prompts' / metadata_key / 'manifest.json').read_text())
    assert metadata['token_split_plan'] == dict(prefix_tokens=prefix, full_prompt_tokens=length)
    result = dict(time=time.time(), evidence='CORRECTNESS_ONLY', request_id=row['request_id'],
        input_tokens=length, prefix_tokens=prefix, suffix_tokens=length-prefix,
        actual_batch='Single request on P, D continuation and subsequent decode',
        fields=fields, input_checks=input_checks, layers=layers,
        strict_greedy_tokens=teacher_result['token_ids'] == token_result['token_ids'],
        output_tokens_teacher=teacher_result['output_tokens'], output_tokens_token=token_result['output_tokens'],
        teacher=teacher_run, token=token_run, original_image_consumer_plan=metadata['image_consumer_plan'],
        scope='Chunk-aligned complete unsplit E+PD teacher, all28 KV layers, original MRoPE/embeddings. '
              'CPU tensor captures invalidate all timing. No active-decode insertion or serving speedup is measured.')
    result['strict_pass'] = all(fields.values()) and all(x['strict'] for x in input_checks) and \
        all(x['imported_prefix_strict'] and x['P_teacher_prefix_strict'] and x['full_prompt_strict'] for x in layers) and \
        result['strict_greedy_tokens']
    write(ROOT / 'analysis' / f'{token_run}_comparison.json', result)
    assert all(x['imported_prefix_strict'] for x in layers), 'The actual NIXL prefix is not byte-exact'
    assert result['strict_pass'], 'Inspect the same minimal correctness case. Do not start performance measurements.'
    print('TOKEN_EXACT_CASE_PASS', token_run, length, prefix, result['output_tokens_token'], flush=True)
    return result


def main():
    import argparse
    global TAG, EC
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default=TAG)
    parser.add_argument('--eager',action='store_true',help='Isolated original eager P/D correctness only, not serving performance')
    parser.add_argument('--fixed-reduction',action='store_true')
    parser.add_argument('--deterministic-reductions',action='store_true')
    parser.add_argument('--wait-for-tag')
    args = parser.parse_args()
    assert args.tag.isidentifier() and args.tag.startswith(('token_CORRECTNESS_C15','token_CORRECTNESS_C30'))
    assert not args.eager or not (args.fixed_reduction or args.deterministic_reductions)
    assert not args.deterministic_reductions or args.fixed_reduction
    assert args.wait_for_tag is None or args.wait_for_tag.isidentifier()
    TAG = args.tag
    EC = Path('/dev/shm') / f'mllm_epd_20261124_{TAG}'
    preserve()
    plan_file = ROOT / 'configs' / f'{TAG}.json'
    assert not plan_file.exists()
    archive = snapshot(TAG)
    manifest = ROOT / 'data/calibration_manifest_template_v2.json'
    rows = json.loads(manifest.read_text())['requests']
    first = rows[0]
    a, b = first['images'][0]['visual_span']
    cases = [('N1_inside_single_image', 0, a + (b-a)//2)]
    sample = rows[6]
    a, b = sample['images'][0]['visual_span']
    cases += [('before_visual', 6, a), ('inside_visual', 6, a+(b-a)//2),
              ('after_visual', 6, b), ('between_images', 6, sample['images'][1]['visual_span'][0]),
              ('text_tail', 6, sample['total_input_tokens']-8)]
    long = rows[31]
    a, b = long['images'][len(long['images'])//2]['visual_span']
    cases.append(('long_document_inside_image', 31, a+(b-a)//2))
    cases += [(f'N8_{index}', index, select_prefix(rows[index]['total_input_tokens'], .2))
              for index in [0, 1, 4, 6, 12, 17, 20, 31]]
    write(plan_file, dict(time=time.time(), status='FROZEN_BEFORE_GPU_TOKEN_RESULTS',
        source_archive_SHA256=archive['archive_SHA256'], manifest_SHA256=digest(manifest), cases=cases,
        GPU_group=[4, 5, 6], evidence='CORRECTNESS_ONLY', output='Natural EOS cap2048, greedy original BF16 BI0',
        compiler_contract=('Explicit original eager P/D diagnostic' if args.eager else
            'Common compiled deterministic reduction control, not stock policy' if args.deterministic_reductions else
            'Unchanged stock compiled P/D'),
        fixed_reduction=args.fixed_reduction,deterministic_reductions=args.deterministic_reductions,
        wait_for_tag=args.wait_for_tag,
        reference='Complete original E+PD execution. Only its prefill chunk boundary is aligned for strict comparison.',
        safety='Wait for all C14 baseline starts to finish. Run one request at a time with no other active decode.',
        gates='Stop on first strict correctness failure. Preserve errors; do not start any token performance arm.',
        observer='Full CPU KV/hidden/embedding copies. Timings cannot become isolated costs or C3 speedups.',
        full_task_quality='PENDING; this is not the official full task-quality gate.'))
    names = []
    teacher_proxy = None
    token_proxy = None
    results = []
    try:
        if args.deterministic_reductions:
            prior=ROOT/'analysis/token_NUMERICAL_C29R1.json'
            proof=json.loads(prior.read_text())
            assert all(all(r['strict'] for r in comparison['KV']) for comparison in proof['comparisons'])
            assert all(r['P_to_D_prefix_strict'] for r in proof['direct_checks'])
            assert len({tuple(v) for v in proof['greedy_tokens'].values()})==1
            write(ROOT/'state'/f'{TAG}_prior_numerical_gate.json',dict(passed=True,source_SHA256=digest(prior),
                scope='Matched N1 compiled control only. Not C3 or full task quality.'))
        if args.wait_for_tag:
            while True:
                leases=[json.loads((ROOT/'state'/f'server_{args.wait_for_tag}_{role}{gpu}.json').read_text())
                    for role,gpu in [('E',4),('P',5),('D',6)]]
                if all(r['finished'] and not Path('/proc',str(r['guardian'])).exists() for r in leases):break
                for lease in leases:
                    proc=Path('/proc',str(lease['guardian']))
                    if not lease['finished']:assert proc.exists() and proc.stat().st_uid==os.getuid()
                assert time.time()<experiment_deadline();time.sleep(1)
        wait_baselines()
        launch_models(names,diagnostic_eager=args.eager,fixed_reduction=args.fixed_reduction,
            deterministic_reductions=args.deterministic_reductions)
        teacher_proxy, teacher_url = launch_proxy(TAG + '_teacher_proxy', 'e_pd', 18848, names)
        for case, index, prefix in cases:
            assert time.time() < experiment_deadline()
            row = rows[index]
            teacher_run = f'{TAG}_{case}_TEACHER'
            token_run = f'{TAG}_{case}_TOKEN'
            teacher_result = capture(row, teacher_run, teacher_url, [(18841, 'P')], prefix)
            token_proxy, token_url = launch_proxy(token_run + '_proxy', 'epd', 18849, names, prefix)
            try:
                token_result = capture(row, token_run, token_url, [(18841, 'P'), (18842, 'D')])
            finally:
                stop_proxy(token_proxy)
                token_proxy = None
            results.append(compare(row, prefix, teacher_run, token_run, teacher_result, token_result))
        write(ROOT / 'analysis' / f'{TAG}_summary.json', dict(time=time.time(), status='CORRECTNESS_PASS',
            cases=len(results), evidence='CORRECTNESS_ONLY', strict_cases=sum(r['strict_pass'] for r in results),
            case_names=[r['token'] for r in results], online_performance='NOT_RUN', full_task_quality='PENDING'))
    except BaseException as error:
        write(ROOT / 'state' / f'{TAG}_failure.json', dict(time=time.time(), status='FAILED_EXECUTION',
            completed_cases=len(results), error=type(error).__name__ + ': ' + str(error)))
        raise
    finally:
        if token_proxy is not None:
            stop_proxy(token_proxy)
        if teacher_proxy is not None:
            stop_proxy(teacher_proxy)
        if names:
            retire([name for name in names if '_E4' not in name])
            retire([name for name in names if '_E4' in name])
    preserve()


if __name__ == '__main__':
    main()
