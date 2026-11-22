"""Qualify the host-setup intervention before any latency A/B."""
import argparse
import json
import time
import torch
from .common import ROOT, write
from .experiment import create
from .freeze_capsule import freeze
from .run import execute
from .qualification import audit, intervals_cover
from .qualification_gate import compare, KV_equal
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .quiescent_admission_qualification import strict_first


def native_state(name, teacher=None):
    folder = ROOT / 'raw' / name
    source = json.loads((folder / 'native_valid_prefix_prefill.json').read_text())['requests']
    dest = json.loads((folder / 'native_valid_prefix_decode.json').read_text())['requests']
    full = json.loads((folder / 'prompt_state_decode.json').read_text())['requests']
    records = {r['sample_id']: r for r in map(json.loads, (folder / 'requests.jsonl').read_text().splitlines())}
    for i, row in records.items():
        rid = 'q' + str(i)
        assert source[rid]['valid_prefix_tokens'] == dest[rid]['valid_prefix_tokens'] == row['input_tokens'] - 1
        assert KV_equal(source[rid]['KV'], dest[rid]['KV'])
        assert len(full[rid]['KV']) == 36 and full[rid]['query_coverage_end'] == row['input_tokens']
        if teacher:
            tf = ROOT / 'raw' / teacher
            gold = json.loads((tf / 'prompt_state_decode.json').read_text())['requests'][rid]
            assert KV_equal(gold['KV'], full[rid]['KV'])
            h = torch.load(full[rid]['hidden_tensor_file'], map_location='cpu', weights_only=True)['hidden_last']
            g = torch.load(gold['hidden_tensor_file'], map_location='cpu', weights_only=True)['hidden_last']
            assert torch.equal(h.flatten(), g.flatten())
            output = {r['sample_id']: r for r in map(json.loads, (tf / 'requests.jsonl').read_text().splitlines())}
            assert row['output_ids'] == output[i]['output_ids']
    for role in ('prefill', 'decode'):
        assert json.loads((folder / f'KV_lifetime_{role}.json').read_text())['passed']
    return dict(passed=True, n=len(records), valid_prefix_exact=True,
                full_prompt_states_observed=True, strict_teacher=teacher)


def assemble():
    proofs = []
    for variant in ('Native', 'Layer', 'StockToken', 'Hybrid'):
        path = ROOT / 'state' / f'D_remote_setup_{variant}_gate_v1.json'
        if not path.exists():
            return False
        value = json.loads(path.read_text())
        if not value.get('passed'):
            return False
        proofs.append(value)
    write(ROOT / 'state/D_remote_setup_deferral_gate_v1.json', dict(passed=True,
        full400_passed=True, proofs=proofs, evidence='CORRECTNESS_ONLY',
        scope='Each variant passed same-geometry N1/N8 and natural mixed32/400 transport/lifetime checks. '
            'Normal completion only. No cancellation/failure-recovery or performance guarantee.'))
    return True


def same_axis_state(teacher, actual):
    expected = ROOT / 'raw' / teacher
    measured = ROOT / 'raw' / actual
    gold = json.loads((expected / 'axis_events_decode.json').read_text())['states']
    state = json.loads((measured / 'axis_events_decode.json').read_text())['states']
    gg = {r['sample_id']: r for r in map(json.loads, (expected / 'requests.jsonl').read_text().splitlines())}
    aa = {r['sample_id']: r for r in map(json.loads, (measured / 'requests.jsonl').read_text().splitlines())}
    rows = []
    for i in aa:
        rid = 'q' + str(i)
        g = torch.load(gold[rid]['tensor_file'], map_location='cpu', weights_only=False)['hidden'][-1]
        a = torch.load(state[rid]['tensor_file'], map_location='cpu', weights_only=False)['hidden'][-1]
        rows.append(dict(request_id=rid, KV_exact=KV_equal(gold[rid]['KV'], state[rid]['KV']),
            hidden_exact=torch.equal(g.flatten(), a.flatten()), greedy_exact=gg[i]['output_ids'] == aa[i]['output_ids']))
    return dict(passed=all(all(r[k] for k in ('KV_exact', 'hidden_exact', 'greedy_exact')) for r in rows),
        rows=rows, same_source_policy_control=teacher,
        scope='Isolate the READ-setup intervention from the unchanged stock-token model path. '
            'Retain the independent original-model Q13 proof and the failed historical cross-source comparison separately.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lane', type=int, choices=range(4))
    args = parser.parse_args()
    torch.set_num_threads(1)
    variant = ('Native', 'Layer', 'StockToken', 'Hybrid')[args.lane]
    axis = 'Token' if variant == 'StockToken' else variant
    first = (0, 2, 4, 6)[args.lane]
    capsule, sha = freeze('remote_setup_deferral_qualification_v1_20261121')
    extras = dict(token_stock_model=True) if variant == 'StockToken' else {}
    if axis == 'Hybrid':
        extras.update(hybrid_suffix_fraction=.125, partition_aligned_chunks=True)
    common = dict(pair=(first, first + 1), cut=0 if axis == 'Native' else 4,
        d_compute_throttle=True, defer_D_remote_setup=True, correctness=True,
        execution_family='INSTALLED_DEFAULT_COMPILE' if variant in ('Native', 'StockToken') else 'MATCHED_NO_COMPILE',
        continuous_source_credit_limit=0 if variant in ('Native', 'StockToken') else 4,
        source_capsule=capsule, source_capsule_SHA256=sha, wall_timeout=4500, serving_timeout=4000, **extras)
    teacher = {'Native': 'Q22_NativeValidPrefix_N8_FullState_v2',
        'Layer': 'Q4_Full36_Teacher_N8', 'StockToken': 'Q13_StockFull36_Teacher4_N8_v1',
        'Hybrid': 'Q4_HybridBoundaryTeacher_f0125_N8'}[variant]
    results = []
    for n in (1, 8, 32, 400):
        dataset = 'qualification8.json' if n <= 8 else 'continuous_qualification32_natural.json' if n == 32 else 'longbench_balanced400_v1.json'
        extra = dict(count=n)
        if n <= 8:
            extra.update(controlled=True, serial=True, output=16, fixed=True)
            # A controlled job must not also advertise continuous source admission.
            extra['continuous_source_credit_limit'] = 0
        if axis == 'Native':
            extra.update(state_capture=True, axis_state_capture=False, gpu_state_digest=True)
        else:
            extra.update(axis_state_capture=True, axis_hash_workers=8,
                axis_capture_hidden_last_only=n > 8, axis_import_hash_after_model=True,
                gpu_state_digest=variant == 'StockToken' or n > 8)
        settings = dict(common, **extra)
        if axis == 'Native' and n <= 8:
            # The historical teacher differs before D in its P-exported KV.
            # Keep that failure. Use a new same-source/pair control to isolate this intervention.
            teacher_settings = dict(settings, defer_D_remote_setup=False)
            teacher = create(f'Q28_NativeControl_SameSource_N{n}_v2', axis, dataset,
                qps=2.8, seed=2026112201, **teacher_settings)
            assert execute(teacher), teacher
        if variant == 'StockToken' and n <= 8:
            teacher_settings = dict(settings, defer_D_remote_setup=False)
            teacher = create(f'Q28_StockTokenControl_SameSource_N{n}_v2', axis, dataset,
                qps=2.8, seed=2026112201, **teacher_settings)
            assert execute(teacher), teacher
        name = create(f'Q28_RemoteSetup_{variant}_N{n}_v1', axis, dataset,
            qps=2.8, seed=2026112201, **settings)
        assert execute(name), name
        folder = ROOT / 'raw' / name
        receipt = json.loads((folder / 'D_remote_setup_deferral.json').read_text())
        assert receipt['passed'] and not receipt['pending']
        if n > 8:
            assert any(r.get('decode_active') and r.get('received') for r in receipt['rows']), 'Deferral path was not exercised'
        if axis == 'Native':
            state = native_state(name, teacher if n <= 8 else None)
        else:
            state = audit(name)
            assert state['passed'] and state['decode_full36_observation_complete']
            assert all(v['status'] == 'PASS' for v in state['imported_prefix_KV'])
        if n <= 8 and variant == 'StockToken':
            numerical = same_axis_state(teacher, name)
        elif n <= 8 and axis != 'Native':
            if n == 1:
                row = strict_first(teacher, name)
                numerical = dict(passed=all(row.values()), rows=[row])
            else:
                numerical = compare(teacher, name)
        elif n <= 8:
            numerical = dict(passed=True, native_strict_teacher=teacher)
        else:
            numerical = quality(name)
        assert numerical['passed'], (name, numerical)
        mix = no_mixed(name)
        results.append(dict(name=name, state=state, numerical=numerical, compute_deferral=mix,
            deferred_metadata_batches=sum(bool(r.get('decode_active') and r.get('received')) for r in receipt['rows'])))
        write(ROOT / 'state' / f'D_remote_setup_{variant}_gate_v1.json', dict(time=time.time(),
            passed=len(results) == 4, full400_passed=len(results) == 4, results=results,
            evidence='CORRECTNESS_ONLY', numerical_family=variant))
    assemble()
    print('REMOTE_SETUP_QUALIFIED', variant, flush=True)


if __name__ == '__main__':
    main()
