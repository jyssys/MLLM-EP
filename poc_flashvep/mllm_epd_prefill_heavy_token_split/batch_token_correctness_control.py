"""Concurrent original-engine token handoffs with one first output per request."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import time

from . import token_correctness_control as single
from .client import payload
from .common import ROOT, digest, experiment_deadline, preserve, write
from .frontend_supply_control import retire
from .ipc_ec_control import ownership_empty
from .llm_probe_control import rpc
from .source_snapshot import snapshot
from .token_split import PLAN_KEY, POLICY_KEY, select_prefix


def headers(base, stage):
    return {'x-request-id': base + ':' + stage,
        'x-campaign-metadata-key': hashlib.sha256(base.encode()).hexdigest(),
        'x-campaign-metadata-mode': 'grid_only'}


async def post(session, port, path, body, request_headers=None):
    async with session.post(f'http://127.0.0.1:{port}' + path, json=body,
                            headers=request_headers) as response:
        result = await response.json()
        assert response.status == 200, (port, response.status, str(result)[:2500])
        return result


async def prepare(session, row, base, prefix):
    body = copy.deepcopy(payload(row, base, stream=False))
    body['max_tokens'] = 1
    encoded = copy.deepcopy(body)
    encoded['kv_transfer_params'] = {POLICY_KEY: dict(suffix_fraction=.2, fixed_prefix=0)}
    encoder = await post(session, 18840, '/v1/chat/completions', encoded, headers(base, 'E:full'))
    assert encoder['usage']['prompt_tokens'] == row['total_input_tokens']
    plan = dict(prefix_tokens=prefix, full_prompt_tokens=row['total_input_tokens'])
    source = copy.deepcopy(body)
    source['kv_transfer_params'] = dict(do_remote_decode=True, do_remote_prefill=False,
        remote_engine_id=None, remote_block_ids=None, remote_host=None, remote_port=None,
        **{PLAN_KEY: plan})
    producer = await post(session, 18841, '/v1/chat/completions', source, headers(base, 'P'))
    params = producer['kv_transfer_params']
    assert params['remote_num_tokens'] == prefix and params['remote_block_ids']
    assert params['do_remote_prefill'] and not params['do_remote_decode']
    continuation = copy.deepcopy(body)
    continuation['kv_transfer_params'] = dict(params, **{PLAN_KEY: plan})
    return dict(base=base, row_id=row['request_id'], prefix=prefix, continuation=continuation,
        encoder_usage=encoder['usage'], producer_usage=producer['usage'],
        original_images=len(row['images']))


async def finish(session, record):
    output = await post(session, 18842, '/v1/chat/completions', record['continuation'],
                        headers(record['base'], 'D'))
    assert output['usage']['completion_tokens'] == 1
    assert output['usage']['prompt_tokens'] == record['continuation']['kv_transfer_params'][PLAN_KEY]['full_prompt_tokens']
    assert len(output['choices'][0]['token_ids']) == 1
    key = hashlib.sha256(record['base'].encode()).hexdigest()
    released = await post(session, 18840, '/campaign/ec_ipc/release', dict(key=key))
    assert len(released) == 1
    return dict(record, response=output, released=released)


def validate(run, contracts):
    records = {role: json.loads((ROOT / 'raw/batch_prefix' / f'{run}_{role}/manifest.json').read_text())
               for role in ['P', 'D']}
    checks = []
    for base, contract in contracts.items():
        p, d = [records[role]['snapshots'][base] for role in ['P', 'D']]
        assert len(p['layers']) == len(d['layers']) == 28
        check = dict(base=base, prefix_tokens=contract['prefix_tokens'],
            full_prompt_tokens=contract['full_prompt_tokens'],
            prefix_ids_strict=p['prefix_ids'] == d['prefix_ids'],
            prefix_mrope_strict=p['prefix_mrope'] == d['prefix_mrope'],
            imported_layers_strict=[a == b for a, b in zip(p['layers'], d['layers'])], query_coverage={})
        for role, start, end in [('P', 0, contract['prefix_tokens']),
                                 ('D', contract['prefix_tokens'], contract['full_prompt_tokens'])]:
            cursor = start
            for event in records[role]['events']:
                for row in event['requests']:
                    if row['base'] != base:
                        continue
                    assert row['start'] == cursor, ('Repeated or missing query rows', base, role, row, cursor)
                    cursor += row['query_tokens']
                    assert row['positions'] == row['expected_positions']
            check['query_coverage'][role] = cursor == end
        check['passed'] = check['prefix_ids_strict'] and check['prefix_mrope_strict'] and \
            all(check['imported_layers_strict']) and all(check['query_coverage'].values())
        checks.append(check)
    result = dict(time=time.time(), run=run, evidence='CORRECTNESS_ONLY', checks=checks,
        passed=all(row['passed'] for row in checks),
        actual_batch_histograms={role: {str(size): sum(e['actual_batch'] == size for e in record['events'])
            for size in sorted({e['actual_batch'] for e in record['events']})} for role, record in records.items()},
        scope='Actual concurrent continuous engines, exact all28 prefix import and original query MRoPE. '
            'One output per request, no subsequent decode. Not a throughput, full suffix KV equivalence, or task-quality result.')
    write(ROOT / 'analysis' / f'{run}_comparison.json', result)
    assert result['passed'], 'Do not run serving performance after a prefix or query-state failure'
    return result


async def cohort(run, rows):
    import aiohttp
    contracts = {f'{run}_{index:04d}': dict(prefix_tokens=select_prefix(row['total_input_tokens'], .2),
        full_prompt_tokens=row['total_input_tokens']) for index, row in enumerate(rows)}
    opened = []
    try:
        for port, role in [(18841, 'P'), (18842, 'D')]:
            rpc(port, 'batch_prefix/begin', dict(name=run + '_' + role, contracts=contracts))
            opened.append((port, role))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), trust_env=False) as session:
            prepared = await asyncio.gather(*(prepare(session, row, base, contracts[base]['prefix_tokens'])
                for row, base in zip(rows, contracts)))
            # This explicit barrier tests independent pending KV lifetimes. It is
            # not a proposed serving admission policy or a measured C3 speedup.
            assert len(prepared) == len(rows)
            values = await asyncio.gather(*(finish(session, record) for record in prepared))
            write(ROOT / 'raw' / f'{run}_responses.json', dict(evidence='CORRECTNESS_ONLY', rows=values))
    finally:
        for port, role in reversed(opened):
            write(ROOT / 'state' / f'{run}_{role}_probe_close.json', rpc(port, 'batch_prefix/end', {}))
    ownership_empty(run, 'cuda_ipc')
    result = validate(run, contracts)
    print('BATCH_TOKEN_PREFIX_PASS', run, result['actual_batch_histograms'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='token_BATCH_C31')
    args = parser.parse_args()
    assert args.tag.isidentifier() and args.tag.startswith('token_BATCH_C31')
    preserve()
    assert time.time() < experiment_deadline()
    prior = ROOT / 'analysis/token_CORRECTNESS_C30R1_summary.json'
    value = json.loads(prior.read_text())
    assert value['status'] == 'CORRECTNESS_PASS' and value['cases'] == value['strict_cases'] == 15
    for role, gpu in [('E', 4), ('P', 5), ('D', 6)]:
        lease = json.loads((ROOT / 'state' / f'server_token_CORRECTNESS_C30R1_{role}{gpu}.json').read_text())
        assert lease['finished'] and not Path('/proc', str(lease['guardian'])).exists()
    manifest = ROOT / 'data/calibration_manifest_template_v2.json'
    pool = json.loads(manifest.read_text())['requests']
    rows = [pool[index] for index in [0, 1, 4, 6, 12, 17, 20, 31]]
    plan = ROOT / 'configs' / f'{args.tag}.json'
    assert not plan.exists()
    archive = snapshot(args.tag)
    write(plan, dict(time=time.time(), evidence='CORRECTNESS_ONLY', prior_SHA256=digest(prior),
        source_archive_SHA256=archive['archive_SHA256'], manifest_SHA256=digest(manifest), GPU_group=[4, 5, 6],
        request_ids=[r['request_id'] for r in rows], suffix_fraction=.2, output_cap=1, EOS=True,
        numerical_policy='Common compiled deterministic reductions, BI0. Stock default remains unchanged.',
        safety='No subsequent decode. Concurrent P first, then concurrent D. CPU copies invalidate timing.',
        untouched='Original images, processor, model forward, attention, layer coverage and engine scheduler.',
        quality='PENDING; one generated token does not establish answer quality.'))
    single.TAG = args.tag
    single.EC = Path('/dev/shm') / f'mllm_epd_20261124_{args.tag}'
    names = []
    results = []
    try:
        single.launch_models(names, fixed_reduction=True, deterministic_reductions=True)
        results.append(asyncio.run(cohort(args.tag + '_N1', rows[:1])))
        results.append(asyncio.run(cohort(args.tag + '_N8', rows)))
        write(ROOT / 'analysis' / f'{args.tag}_summary.json', dict(time=time.time(),
            evidence='CORRECTNESS_ONLY', passed=all(r['passed'] for r in results), cohorts=results,
            task_quality='PENDING', active_decode_insertion='NOT_RUN', online_speedup='NOT_RUN'))
    except BaseException as error:
        write(ROOT / 'state' / f'{args.tag}_failure.json', dict(time=time.time(), status='FAILED_EXECUTION',
            error=type(error).__name__ + ': ' + str(error), completed_cohorts=len(results)))
        raise
    finally:
        if names:
            retire([name for name in names if '_E4' not in name])
            retire([name for name in names if '_E4' in name])
    preserve()


if __name__ == '__main__':
    main()
