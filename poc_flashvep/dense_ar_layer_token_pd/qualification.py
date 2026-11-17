"""Escalate exact token-axis execution only after the preceding state gate."""
import fcntl
import json
from .common import ROOT, SLO_PRIOR, QUIET_LOCK, resources, save, write
from .run import execute


def configuration(name, count=8, suffix=None, chunk=None):
    from ..duetserve_agg_decode import configuration as inherited
    inherited.ROOT, inherited.PRIOR = ROOT, SLO_PRIOR
    cfg = inherited.base(name, arm='PD_G1', rate=2., count=count, cap=16,
        seed=2026111702, reuse_prefill_token=False, axis_state_witness=True,
        clean_timing=False, correctness_only=True, ar_cudagraph_mode='NONE',
        warmup_batch_sizes=[], warmup_count=0,
        entry_module='poc_flashvep.dense_ar_layer_token_pd.serving', runtime_observation=True)
    if suffix is not None:
        cfg.update(token_axis=True, token_suffix=suffix)
    if chunk is not None:
        cfg.update(chunk=chunk, atomic_prompt=False)
    return cfg


def rows(folder):
    return {r['sample_id']: r for r in map(json.loads, (folder / 'requests.jsonl').read_text().splitlines())}


def gate(name, reference):
    import torch
    torch.set_num_threads(1)
    folder, ref = [ROOT / 'raw' / n for n in (name, reference)]
    cfg = json.loads((folder / 'configuration.json').read_text())
    source = json.loads((folder / 'axis_state_prefill.json').read_text())
    destination = json.loads((folder / 'axis_state_decode.json').read_text())
    expected = json.loads((ref / 'axis_state_prefill.json').read_text())['requests']
    actual, reference_rows = rows(folder), rows(ref)
    from ..duetserve_agg_decode import analyze as original_outputs
    original_outputs.ROOT = ROOT
    original_gate = original_outputs.output_gate(name)
    checks = {}
    for key, state in destination['requests'].items():
        index = int(key[1:])
        hidden = torch.load(folder / f'axis_hidden_prefill_{key}.pt', weights_only=True)
        suffix_hidden = torch.load(folder / f'axis_hidden_decode_{key}.pt', weights_only=True)
        if cfg.get('token_axis'):
            hidden = torch.cat((hidden, suffix_hidden))
        expected_hidden = torch.load(ref / f'axis_hidden_prefill_{key}.pt', weights_only=True)
        from .token_axis import suffix_for
        query_start = cfg['length'] - suffix_for(cfg, key) if cfg.get('token_axis') else cfg['length'] - 1
        reference_output = reference_rows[index]['output_ids']
        actual_output = actual[index]['output_ids']
        checks[key] = dict(output=actual_output[:min(len(reference_output), len(actual_output))] ==
            reference_output[:min(len(reference_output), len(actual_output))],
            full_prompt_KV=state['KV'] == expected[key]['KV'],
            final_hidden=bool(torch.equal(hidden.contiguous().view(torch.uint8),
                                          expected_hidden.contiguous().view(torch.uint8))),
            P_end=source['requests'][key]['query_end'], D_start=state['first_query'], D_end=state['query_end'],
            D_expected_start=query_start, query_partition=state['first_query'] == query_start and
                (not cfg.get('token_axis') or source['requests'][key]['query_end'] == query_start))
    graph_definitions = []
    for path in folder.glob('extension_*.jsonl'):
        for row in map(json.loads, path.read_text().splitlines()):
            if row.get('event') == 'CAPTURED_FULL_GRAPH_LAYERS':
                graph_definitions.append(row)
    graphs_valid = bool(graph_definitions) and all(r['passed'] and r['actual'] == list(range(36)) for r in graph_definitions)
    coverage = all((row.get('graph_replay') and graphs_valid) or row['layers'] == list(range(36))
                   for doc in (source, destination) for row in doc['visits'])
    lifetimes = {p.name: json.loads(p.read_text()).get('passed') for p in folder.glob('KV_lifetime_*.json')}
    passed = (original_gate['passed'] and len(checks) == cfg['count'] and coverage and len(lifetimes) == 2 and all(lifetimes.values())
        and all(c['output'] and c['full_prompt_KV'] and c['final_hidden'] and c['query_partition'] for c in checks.values()))
    result = dict(name=name, reference=reference, passed=passed, checks=checks,
        full36_every_model_call=coverage, captured_full_graph_definitions=len(graph_definitions),
        original_output_reference_gate=original_gate, normal_KV_lifetime=lifetimes,
        evidence='CORRECTNESS_ONLY', scope='Exact output and all valid36-layer prompt KV. Concatenate only actually computed P/D hidden rows. The Native last-query replay is accounted separately. CPU copies perturb timing.')
    write(ROOT / 'analysis' / (name + '_state_gate.json'), result)
    print('STATE_GATE', name, passed, flush=True)
    return passed


def job(cfg):
    return execute(save(cfg))


def main():
    try:
        reference8 = 'Native_state_N8_O16_v1'
        qualified = [512]
        assert gate('Token512_state_N8_O16_v2', reference8)
        for suffix in (1024, 2048, 4096):
            name = f'Token{suffix}_state_N8_O16_v1'
            if job(configuration(name, suffix=suffix)) and gate(name, reference8):
                qualified.append(suffix)
        reference32 = 'Native_state_N32_O16_v1'
        assert job(configuration(reference32, count=32))
        assert gate(reference32, reference32)
        for suffix in qualified:
            name = f'Token{suffix}_state_N32_O16_v1'
            if job(configuration(name, count=32, suffix=suffix)):
                gate(name, reference32)
        name = 'P_chunk512_state_N32_O16_v1'
        if job(configuration(name, count=32, chunk=512)):
            gate(name, reference32)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
