"""Check unseen outputs before reading any unseen performance result."""
import fcntl
import json
from .common import ROOT, SLO_PRIOR, QUIET_LOCK, resources, save, write
from .run import execute
from .analysis import records

REFERENCE = 'unseen_Native_reference_N128_O200_repeat1_v1'


def main():
    from ..duetserve_agg_decode import configuration
    configuration.ROOT, configuration.PRIOR = ROOT, SLO_PRIOR
    names = []
    try:
        for repeat in (1, 2):
            name = f'unseen_Native_reference_N128_O200_repeat{repeat}_v1'
            cfg = configuration.base(name, arm='PD_G1', count=128, cap=200, rate=3.,
                seed=2026111720, dataset='heldout128_8000.json',
                entry_module='poc_flashvep.dense_ar_layer_token_pd.serving',
                correctness_only=True, clean_timing=False,
                full_prompt_state_witness=repeat == 1,
                full_prompt_state_witness_roles=['prefill'])
            assert execute(save(cfg)), name
            names.append(name)
        name = 'unseen_independent_DP2_FULL_B8192_S1024_N32_O200_v1'
        cfg = configuration.base(name, budget=8192, slots=1024, count=32, cap=200,
            rate=2., seed=2026111720, dataset='heldout128_8000.json',
            entry_module='poc_flashvep.dense_ar_layer_token_pd.serving',
            correctness_only=True, clean_timing=False, full_prompt_state_witness=True,
            post_graph_state_witness=True, full_prompt_state_witness_roles=['replica0', 'replica1'],
            explicit_cudagraph_capture_sizes=configuration.standard_capture_sizes(1024))
        assert execute(save(cfg)), name
        names.append(name)
        reference = {r['sample_id']: r for r in records(ROOT / 'raw' / names[0] / 'requests.jsonl')}
        comparisons = {}
        for candidate in names[1:]:
            rows = records(ROOT / 'raw' / candidate / 'requests.jsonl')
            comparisons[candidate] = dict(count=len(rows), passed=bool(rows) and all(
                r['input_ids'] == reference[r['sample_id']]['input_ids'] and
                r['output_ids'] == reference[r['sample_id']]['output_ids'] for r in rows))
        source = json.loads((ROOT / 'raw' / names[0] / 'full_prompt_state_prefill.json').read_text())['requests']
        states = {}
        for role in ('replica0', 'replica1'):
            for rid, value in json.loads((ROOT / 'raw' / names[2] / f'full_prompt_state_{role}.json').read_text())['requests'].items():
                states[rid] = dict(KV=value['KV'] == source[rid]['KV'],
                    hidden=value['final_hidden_SHA256'] == source[rid]['final_hidden_SHA256'])
        passed = len(reference) == 128 and all(v['passed'] for v in comparisons.values()) and len(states) == 32 and all(all(v.values()) for v in states.values())
        write(ROOT / 'analysis/unseen_reference_gate.json', dict(reference=REFERENCE,
            comparisons=comparisons, states=states, passed=passed, evidence='CORRECTNESS_ONLY',
            scope='Two independent Native starts, plus local Full36 standard continuous DP2 outputs and full prompt KV/hidden on32 requests. These timings cannot select policies or baseline settings.'))
        assert passed
        print('UNSEEN_REFERENCE_PASS', flush=True)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
