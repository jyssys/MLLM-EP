"""Scale state checks without redundant copies of full hidden tensor files."""
import fcntl
import json
from .common import ROOT, QUIET_LOCK, resources, save, write
from .qualification import configuration
from .run import execute
from .fresh_reference import REFERENCE
from ..duetserve_agg_decode.analyze import output_gate


def gate(name, reference=None, reference_state=None):
    folder = ROOT / 'raw' / name
    cfg = json.loads((folder / 'configuration.json').read_text())
    source_path = reference_state or ROOT / 'raw' / reference / 'axis_state_prefill.json'
    expected = json.loads(source_path.read_text())['requests']
    src = json.loads((folder / 'axis_state_prefill.json').read_text())
    dst = json.loads((folder / 'axis_state_decode.json').read_text())
    from .token_axis import suffix_for
    checks = {}
    for rid, value in dst['requests'].items():
        suffix = suffix_for(cfg, rid)
        hashes = expected[rid]['partition_hashes'][str(suffix)]
        checks[rid] = dict(KV=value['KV'] == expected[rid]['KV'],
            P_hidden=src['requests'][rid]['final_hidden_SHA256'] == hashes['prefix_SHA256'],
            D_hidden=value['final_hidden_SHA256'] == hashes['suffix_SHA256'],
            P_end=src['requests'][rid]['query_end'] == 8000-suffix,
            D_start=value['first_query'] == 8000-suffix,
            D_end=value['query_end'] == 8000)
    coverage = all(r['layers'] == list(range(36)) for doc in (src, dst) for r in doc['visits'])
    lifetime = {p.name: json.loads(p.read_text())['passed'] for p in folder.glob('KV_lifetime_*.json')}
    from ..duetserve_agg_decode import analyze
    analyze.ROOT = ROOT
    output = analyze.output_gate(name)
    passed = len(checks) == cfg['count'] and all(all(v.values()) for v in checks.values()) and coverage and len(lifetime)==2 and all(lifetime.values()) and output['passed']
    result = dict(passed=passed, checks=checks, full36=coverage, lifetime=lifetime, output=output,
        evidence='CORRECTNESS_ONLY', scope='SHA256 of exact contiguous BF16 bytes. Compare actual P and D hidden segments with independently computed Native whole-prompt segments. Full valid36-layer destination prompt KV. No full hidden files are needed for this N128 extension. Latencies are not clean performance.')
    write(ROOT / 'analysis' / (name + '_hash_state_gate.json'), result)
    assert passed, result
    return result


def main():
    assert json.loads((ROOT / 'analysis/unseen_reference_gate.json').read_text())['passed']
    names = []
    try:
        name = 'unseen_Native_axis_state_N128_O16_hash_v1'
        cfg = configuration(name, count=128)
        cfg.update(dataset='heldout128_8000.json', reference_name=REFERENCE,
            axis_state_save_tensors=False, witness_hidden_partition_tokens=[2048,2656,2672,4096])
        assert execute(save(cfg))
        for suffix in (2048, 2672, 4096):
            candidate = f'unseen_Token{suffix}_axis_state_N128_O16_hash_v1'
            cfg = configuration(candidate, count=128, suffix=suffix)
            cfg.update(dataset='heldout128_8000.json', reference_name=REFERENCE,
                       axis_state_save_tensors=False)
            assert execute(save(cfg))
            gate(candidate, name)
            names.append(candidate)
        write(ROOT / 'analysis/Token_N128_state_qualification.json', dict(passed=True,
            names=names, reference=name, evidence='CORRECTNESS_ONLY'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
