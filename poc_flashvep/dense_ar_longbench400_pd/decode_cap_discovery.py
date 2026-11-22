"""Separate decode concurrency from transfer setup under the same frozen source."""
import argparse
import json
import time
import torch
from .common import ROOT, write, dispatch_allowed
from .experiment import create
from .freeze_capsule import freeze
from .geometry import matched_suffixes
from .run import execute
from .qualification import audit
from .quality_gate import compare as quality
from .compute_throttle_qualification import no_mixed
from .remote_setup_qualification import assemble, native_state
from .metrics import summarize


METHODS = [dict(label='Native', axis='Native', compiled=True),
           dict(label='Layer8', axis='Layer', cut=8),
           dict(label='StockToken8', axis='Token', cut=8, compiled=True, stock=True),
           dict(label='Hybrid4', axis='Hybrid', cut=4, fraction=.125),
           dict(label='WholeDeflection', axis='Native', compiled=True, deflection=True)]


def extra(action, dataset):
    value = {}
    if action['axis'] == 'Token':
        data = json.loads((ROOT / 'data' / dataset).read_text())
        suffix, work = matched_suffixes([r['input_tokens'] for r in data], action['cut'])
        value.update(token_stock_model=True, suffix_by_request={'q' + str(i): s for i, s in enumerate(suffix)},
            aggregate_work_audit=work, qualification_gate='C3_stock_token_gate_v1.json')
    elif action['axis'] != 'Native':
        value['qualification_gate'] = 'C3_continuous_actions_gate_v1.json'
    if action['axis'] == 'Hybrid':
        value.update(hybrid_suffix_fraction=action['fraction'], partition_aligned_chunks=True)
    if action.get('deflection'):
        value['whole_prefill_deflection'] = True
    return value


def full_native_state(name, deflection):
    if not deflection:
        return native_state(name)
    folder = ROOT / 'raw' / name
    direct = {r['request_id'] for r in map(json.loads, (folder / 'whole_prefill_deflections.jsonl').read_text().splitlines())}
    source = json.loads((folder / 'native_valid_prefix_prefill.json').read_text())['requests']
    dest = json.loads((folder / 'native_valid_prefix_decode.json').read_text())['requests']
    full = json.loads((folder / 'prompt_state_decode.json').read_text())['requests']
    rows = list(map(json.loads, (folder / 'requests.jsonl').read_text().splitlines()))
    assert direct
    for r in rows:
        rid = 'q' + str(r['sample_id'])
        assert len(full[rid]['KV']) == 36 and full[rid]['query_coverage_end'] == r['input_tokens']
        if rid not in direct:
            assert source[rid]['KV'] == dest[rid]['KV']
    for role in ('prefill', 'decode'):
        assert json.loads((folder / f'KV_lifetime_{role}.json').read_text())['passed']
    return dict(passed=True, n=len(rows), actual_deflections=len(direct), Native_imports_exact=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lane', type=int, choices=range(4))
    args = parser.parse_args()
    torch.set_num_threads(1)
    first = 2 * args.lane
    capsule, sha = freeze('decode_concurrency_setup_discovery_v1_20261121')
    cases = [(a, cap) for a in METHODS for cap in (1, 2)]
    selected = cases[args.lane::4]
    write(ROOT / 'configs' / f'decode_concurrency_discovery_lane{args.lane}_v1.json', dict(cases=selected,
        source_capsule_SHA256=sha, B=8192, S=1024, scope='D running admission is a separate explicit limit. '
            'Do not call cap2 an actual batch2 guarantee or an unmodified Native scheduler. '
            'All methods use the same cap in each controlled comparison. Keep uncapped Native and Token baselines. '
            'N32 and N400 correctness precede timing. Retain every safety and latency failure.',
        evidence='FROZEN_CAUSAL_ADMISSION_DESIGN'))
    while not assemble():
        assert dispatch_allowed()
        time.sleep(5)
    for action, cap in selected:
        label = action['label']
        common = dict(cut=action.get('cut', 0), pair=(first, first + 1), d_compute_throttle=True,
            defer_D_remote_setup=True, D_running_admission_cap=cap,
            execution_family='INSTALLED_DEFAULT_COMPILE' if action.get('compiled') else 'MATCHED_NO_COMPILE',
            continuous_source_credit_limit=4 if action.get('deflection') or not action.get('compiled') else 0,
            source_capsule=capsule, source_capsule_SHA256=sha, wall_timeout=4500, serving_timeout=4000)
        proofs = []
        path = ROOT / 'state' / f'D_admission_cap_{cap}_{label}_gate_v1.json'
        for n, dataset in ((32, 'continuous_qualification32_natural.json'), (400, 'longbench_balanced400_v1.json')):
            capture = dict(state_capture=True, axis_state_capture=False) if action['axis'] == 'Native' else dict(
                axis_state_capture=True, axis_hash_workers=8, axis_capture_hidden_last_only=True,
                axis_import_hash_after_model=True)
            name = create(f'Q33_Dcap{cap}_{label}_Natural{n}_v1', action['axis'], dataset,
                qps=2.8, seed=2026112201, correctness=True, gpu_state_digest=True,
                **common, **capture, **extra(action, dataset))
            assert execute(name), name
            folder = ROOT / 'raw' / name
            ready = json.loads((folder / 'actual_runtime_decode.json').read_text())
            assert ready['scheduler_S'] == 1024 and ready['runtime_max_num_running_requests'] == cap
            if action['axis'] == 'Native':
                state = full_native_state(name, action.get('deflection'))
            else:
                state = audit(name)
                assert state['passed'] and state['decode_full36_observation_complete']
                assert all(v['status'] == 'PASS' for v in state['imported_prefix_KV'])
            score = quality(name)
            assert score['passed'], (name, score)
            mix = no_mixed(name)
            assert mix['maximum_D_batch'] <= cap
            proofs.append(dict(name=name, state=state, quality=score, compute_deferral=mix))
            write(path, dict(passed=len(proofs) == 2, full400_passed=len(proofs) == 2,
                action=action, cap=cap, proofs=proofs, evidence='CORRECTNESS_ONLY',
                scope='Inherited original-model/action N8 proofs plus this cap composition Natural32/400 validation. '
                    'No single-start ITL or performance success is implied.'))
        for qps in (1.8, 2.8):
            name = create(f'DcapTune_Discovery200_q{qps}_{label}_Cap{cap}_v1', action['axis'],
                'balanced400_discovery.json', qps=qps, seed=2026112291, discovery_only=True,
                D_cap_qualification_gate=path.name, **common, **extra(action, 'balanced400_discovery.json'))
            assert execute(name), name
            m = summarize(name)
            score = quality(name)
            print('D_CAP_DISCOVERY', label, cap, qps, m['TTFT_p95_ms'], m['ITL_p99_ms'],
                m['TPOT_p95_ms'], m['SLO_qualified_output_tokens_per_second'], score['passed'], flush=True)
    print('D_CAP_DISCOVERY_COMPLETE', args.lane, flush=True)


if __name__ == '__main__':
    main()
