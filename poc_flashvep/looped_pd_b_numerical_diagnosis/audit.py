"""Frozen-trace red test, exact failure manifests, and source provenance."""
import argparse
import json
import subprocess
from collections import Counter
from .common import ROOT, PRIOR, REPO, RLT, MODEL_PINS, setup, write, digest

PAIRS = [
    ('ouro_original_repeat', 'repair_r0_ouro_prefix8_PD_G1_repeat0', 'repair_r0_ouro_prefix8_PD_G1_repeat2'),
    ('ouro_original_B', 'repair_r0_ouro_prefix8_PD_G1_repeat0', 'repair_r0_ouro_prefix8_B_repeat1'),
    ('ouro_residual', 'repair_epoch_ouro_W1_PD_G1_N2_rate1_seed2026107001', 'repair_epoch_ouro_W1_B_N2_rate1_seed2026107001'),
    ('huginn_repeat', 'repair_expand_canonical_all_huginn_N200_cap128_PD_G1_repeat0', 'repair_expand_canonical_all_huginn_N200_cap128_PD_G1_repeat2'),
    ('huginn_B', 'repair_expand_canonical_all_huginn_N200_cap128_PD_G1_repeat0', 'repair_expand_canonical_all_huginn_N200_cap128_B_repeat1'),
]


def rows(name):
    return {r['sample_id']: r for line in (PRIOR/'raw'/name/'requests.jsonl').read_text().splitlines()
            if (r := json.loads(line)) and not r.get('warmup')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assert-parity', action='store_true')
    args = parser.parse_args()
    setup()
    result = []
    cases = []
    for label, left, right in PAIRS:
        a, b = rows(left), rows(right)
        common = sorted(a.keys() & b.keys())
        differences = []
        for i in common:
            if a[i]['output_ids'] == b[i]['output_ids']:
                continue
            pos = next((j for j, (x, y) in enumerate(zip(a[i]['output_ids'], b[i]['output_ids'])) if x != y),
                       min(len(a[i]['output_ids']), len(b[i]['output_ids'])))
            differences.append(dict(sample_id=i, first_divergence=pos))
            cases.append(dict(label=label, left=left, right=right, sample_id=i, first_divergence=pos,
                              input_equal=a[i]['input_ids'] == b[i]['input_ids'],
                              history_equal=a[i]['output_ids'][:pos] == b[i]['output_ids'][:pos],
                              left_row=a[i], right_row=b[i],
                              files={run: dict(requests_sha256=digest(PRIOR/'raw'/run/'requests.jsonl'),
                                               contract=json.loads((PRIOR/'raw'/run/'serving_contract.json').read_text()))
                                     for run in (left, right)}))
        item = dict(label=label, left=left, right=right, N=len(common), exact=len(common)-len(differences),
                    complete=len(common)==len(a)==len(b), differences=differences, evidence='TRACE_ANALYSIS')
        result.append(item)
        print(json.dumps(item), flush=True)
    write(ROOT/'configs/frozen_red_test.json', result)
    write(ROOT/'configs/failure_manifest.json', cases)
    paths = list((REPO/'poc_flashvep/looped_pd_b_slo_serving').glob('*.py'))
    paths += list((REPO/'poc_flashvep/looped_pd_abc').glob('*.py'))
    paths += list((RLT/'vllm_rlt').rglob('*.py'))
    paths += list((REPO/'poc_flashvep/ref_loopserve').rglob('*.py'))
    paths += list((REPO/'poc_flashvep/ref_loopserve').rglob('README.md'))
    source_hashes = {str(p): digest(p) for p in paths}
    lock = ROOT/'configs/read_only_source_lock.json'
    if lock.exists() and json.loads(lock.read_text()) != source_hashes:
        raise RuntimeError('A read-only source changed during diagnosis')
    if not lock.exists():
        write(lock, source_hashes)
    status = subprocess.check_output(['git','status','--porcelain=v1','-uno'], cwd=REPO, text=True)
    write(ROOT/'configs/environment_audit.json', dict(
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO,text=True).strip(),
        status_counts=dict(Counter(line[:2] for line in status.splitlines())),
        model_pins=MODEL_PINS, RLT=str(RLT),
        stage_index_sha256=__import__('hashlib').sha256(subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)).hexdigest()))
    if args.assert_parity and any(r['exact'] != r['N'] or not r['complete'] for r in result):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
