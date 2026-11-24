"""Keep stock Native intact while measuring the new common compiler contract."""
import argparse
import asyncio
import json
from pathlib import Path
import random
import time

from .client import execute, load_replay_counts, resolve_manifest
from .common import ROOT, digest, experiment_deadline, preserve, write
from .frontend_supply_control import retire
from .ipc_ec_control import launch, ownership_empty, stop_proxy
from .llm_probe_control import rpc
from .metrics import summarize
from .numerical import embedding_metrics
from .source_snapshot import snapshot
from .token_correctness_control import capture, launch_proxy


def compare_native_state(values, records):
    import torch
    length = len(values['T']['prompt_token_ids'])
    event = records['D']['snapshot']['event']
    assert event['start'] == length - 1 and event['query_tokens'] == 1
    fields = dict(
        prompt_ids=all(torch.equal(values['T']['prompt_token_ids'], values[r]['prompt_token_ids']) for r in ['P', 'D']),
        mrope=all(torch.equal(values['T']['full_prompt_mrope'], values[r]['full_prompt_mrope']) for r in ['P', 'D']),
        full_delta=len({records[r]['snapshot']['mrope_delta'] for r in ['T', 'P', 'D']}) == 1)
    teacher_event = records['T']['snapshot']['event']
    for field in ['inputs_embeds', 'positions']:
        actual = values['D'][f'{field}_{event["index"]}']
        target = values['T'][f'{field}_{teacher_event["index"]}']
        target = target[-1:] if field != 'positions' else target[..., -1:]
        fields[field + '_last_query'] = torch.equal(actual, target)
    layers = []
    for index in range(28):
        key = f'prompt_KV_{index}'
        teacher, source, destination = [values[r][key] for r in ['T', 'P', 'D']]
        assert teacher.shape == source.shape == destination.shape
        assert teacher.shape[0] == length
        last = embedding_metrics(destination[-1:], teacher[-1:])
        layers.append(dict(layer=index, P_teacher_strict=torch.equal(source, teacher),
            imported_prefix_strict=torch.equal(source[:-1], destination[:-1]),
            full_prompt_strict=torch.equal(destination, teacher), last_query=last,
            last_query_tolerance_pass=last['finite'] and last['relative_L2'] <= .02 and last['cosine'] >= .999))
    return dict(fields=fields, layers=layers,
        passed=all(fields.values()) and all(r['P_teacher_strict'] and r['imported_prefix_strict'] and
            r['last_query_tolerance_pass'] for r in layers),
        scope='Native imports L-1 rows, then recomputes its last prompt query. '
            'Test that row separately. Whole-prompt averaging cannot hide its numerical error.')


def compare_capture(row, teacher_run, native_run, teacher_result, native_result):
    from safetensors.torch import load_file
    values = {}
    records = {}
    for role, run, worker in [('T', teacher_run, 'P'), ('P', native_run, 'P'), ('D', native_run, 'D')]:
        path = ROOT / 'raw/llm_reference' / f'{run}_{worker}/manifest.json'
        records[role] = json.loads(path.read_text())
        record = records[role]
        assert digest(record['tensor_file']) == record['tensor_SHA256']
        values[role] = load_file(record['tensor_file'])
    result = dict(time=time.time(), evidence='CORRECTNESS_ONLY', request_id=row['request_id'],
        teacher=teacher_run, native=native_run, **compare_native_state(values, records),
        strict_greedy_tokens=teacher_result['token_ids'] == native_result['token_ids'],
        teacher_output_tokens=teacher_result['output_tokens'], native_output_tokens=native_result['output_tokens'],
        quality='PENDING; BI0 does not guarantee identical tokens across different execution geometry.')
    write(ROOT / 'analysis' / f'{native_run}_comparison.json', result)
    assert result['passed'], 'NativeCommon state gate failed. Do not time this arm.'
    return result


def set_common_ipc_policy():
    for port in [18841, 18842]:
        rpc(port, 'ec_ipc/ack_policy', dict(mode='file'))
    result = rpc(18840, 'ec_ipc/inline_policy', dict(enabled=True))
    assert result[0]['inline_role'] == 'prefill'


async def measurements(tag):
    path = resolve_manifest(None)
    rows = json.loads(path.read_text())['requests']
    random.Random(2026112404).shuffle(rows)
    reference = ROOT / 'raw/reference_main400_natural_template_v2_v0_requests.jsonl'
    counts = load_replay_counts(reference)
    results = []
    for suffix, outputs in [('warmup', counts), ('natural_q2', None), ('replay_q2', counts)]:
        run = f'{tag}_{suffix}_v0'
        values = await execute(rows, ['http://127.0.0.1:18847'], run, 2., 2026112402, outputs,
            metadata=dict(evidence='DIAGNOSTIC_WARMUP' if suffix == 'warmup' else 'MEASURED_ONLINE',
                manifest_SHA256=digest(path), quality_gate='PENDING', independent_starts=1,
                common_compiler_policy=True, stock_native_unchanged=True,
                compiler=dict(dynamic_scale_rblock=False, deterministic=True,
                    combo_kernels=True, benchmark_combo_kernel=False),
                replay_reference_SHA256=digest(reference) if outputs is not None else None,
                scope='Native only. Original full P and standard NIXL D last-query recomputation. '
                    'No token-split adapter, no GPU observer, no artificial admission cap.'))
        assert len(values) == 100 and all(r['status'] == 'MEASURED_ONLINE' for r in values)
        ownership_empty(run, 'cuda_ipc')
        metrics = summarize(run)
        results.append(metrics)
        print('COMMON_NATIVE_PILOT', run, {k: metrics[k] for k in
            ['TTFT_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms', 'gate_attainment']}, flush=True)
    write(ROOT / 'analysis' / f'{tag}_pilot_summary.json', dict(time=time.time(), results=results,
        independent_starts=1, evidence='MEASURED_ONLINE', task_quality='PENDING',
        confidence_interval='NOT_AVAILABLE', numerical_contract='Common compiled policy, not stock NativeStrong'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='ec_COMMON_C32')
    args = parser.parse_args()
    assert args.tag.isidentifier() and args.tag.startswith('ec_COMMON_C32')
    preserve()
    assert time.time() < experiment_deadline()
    for role, gpu in [('E', 4), ('P', 5), ('D', 6)]:
        previous = json.loads((ROOT / 'state' / f'server_token_BATCH_C31_{role}{gpu}.json').read_text())
        assert previous['finished'] and not Path('/proc', str(previous['guardian'])).exists()
    batch = ROOT / 'analysis/token_BATCH_C31_summary.json'
    assert json.loads(batch.read_text())['passed']
    archive = snapshot(args.tag)
    plan = ROOT / 'configs' / f'{args.tag}.json'
    assert not plan.exists()
    write(plan, dict(time=time.time(), evidence='CORRECTNESS_ONLY then MEASURED_ONLINE',
        source_archive_SHA256=archive['archive_SHA256'], prior_batch_SHA256=digest(batch),
        method='Common-policy Native EPD. No token-split adapter.', GPU_group=[4, 5, 6],
        P_D_compiler=dict(dynamic_scale_rblock=False, deterministic=True, combo_kernels=True,
            benchmark_combo_kernel=False), E_compiler='Unchanged', natural_cap=2048,
        preflight='N1 then eight serial requests. Same-policy full E+P teacher. All28 imported prefix strict.',
        last_query_gate=dict(relative_L2_max=.02, cosine_min=.999),
        numerical_caveat='BI0 teacher and Native D have different last-query geometry. Record greedy differences.',
        SLO='Unchanged TTFT5000ms, all ITL34ms, TPOT33ms. Task quality remains separate.',
        performance='One independent start, q2 Main100 Natural and equal-output replay. No primary GO or CI claim.'))
    names = []
    proxy = None
    teacher_proxy = None
    results = []
    try:
        names, proxy = launch(args.tag, 'cuda_ipc', fixed_reduction=True, deterministic_reductions=True,
            owned_names=names)
        set_common_ipc_policy()
        teacher_proxy, teacher_url = launch_proxy(args.tag + '_teacher_proxy', 'e_pd', 18848, names)
        pool = json.loads((ROOT / 'data/calibration_manifest_template_v2.json').read_text())['requests']
        for case, index in [('N1', 0)] + [(f'N8_{i}', i) for i in [0, 1, 4, 6, 12, 17, 20, 31]]:
            assert time.time() < experiment_deadline()
            row = pool[index]
            teacher_run = f'{args.tag}_{case}_TEACHER'
            native_run = f'{args.tag}_{case}_NATIVE'
            teacher = capture(row, teacher_run, teacher_url, [(18841, 'P')])
            native = capture(row, native_run, 'http://127.0.0.1:18847', [(18841, 'P'), (18842, 'D')])
            results.append(compare_capture(row, teacher_run, native_run, teacher, native))
            print('COMMON_NATIVE_STATE_PASS', case, results[-1]['strict_greedy_tokens'], flush=True)
        stop_proxy(teacher_proxy)
        teacher_proxy = None
        write(ROOT / 'analysis' / f'{args.tag}_correctness.json', dict(time=time.time(),
            passed=all(r['passed'] for r in results), rows=results, task_quality='PENDING'))
        asyncio.run(measurements(args.tag))
    except BaseException as error:
        write(ROOT / 'state' / f'{args.tag}_failure.json', dict(time=time.time(), status='FAILED_EXECUTION',
            error=type(error).__name__ + ': ' + str(error), completed_state_cases=len(results)))
        raise
    finally:
        if teacher_proxy is not None:
            stop_proxy(teacher_proxy)
        if proxy is not None:
            stop_proxy(proxy)
        owned = [name for name in names if (ROOT / 'state' / f'server_{name}.json').exists()]
        if owned:
            retire([name for name in owned if '_E4' not in name])
            retire([name for name in owned if '_E4' in name])
    preserve()


if __name__ == '__main__':
    main()
