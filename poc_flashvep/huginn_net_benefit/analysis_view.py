"""Exact qualification views. A short-output discovery keeps the full prompt work."""
from .common import *
from ..completion_aware_loop_pd import analyze as parent


def metrics(name):
    configure()
    return parent.metrics(name)


def qualify_frozen_reference(name):
    configure()
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    if not cfg.get('qualification_output_prefix'):
        return parent.qualify_frozen_reference(name)
    # This branch changes the requested output cap, not recurrent prompt work.
    # It compares every requested token and the complete valid prompt state.
    cap = cfg['cap']
    assert type(cap) is int and 0 < cap < 16
    assert cfg['request_caps'] == [cap]*cfg['count'] and cfg['model'] == 'huginn'
    result = parent.qualify(name)
    if not result['complete']:
        return result
    folder = ROOT/'raw'/name
    reference = parent.OLD/'raw'/'n32_huginn_native_full_state32'
    expected = set(range(cfg['seed'], cfg['seed']+cfg['count']))
    def outputs(path):
        return {r['seed']:r for r in map(json.loads, (path/'requests.jsonl').read_text().splitlines())
            if not r.get('warmup') and r['seed'] in expected}
    def states(path):
        values = {}
        for path in path.glob('prompt_states_*.jsonl'):
            for row in map(json.loads, path.read_text().splitlines()):
                if row['seed'] not in expected or row['prompt_tokens'] != cfg['length']:
                    continue
                value = (row['hidden_sha256'], row['planes'])
                assert row['seed'] not in values or values[row['seed']] == value
                values[row['seed']] = value
        return values
    before, after = outputs(reference), outputs(folder)
    original, actual = states(reference), states(folder)
    bad_output = [seed for seed in expected if seed not in before or seed not in after
        or before[seed]['input_ids'] != after[seed]['input_ids']
        or len(before[seed]['output_ids']) < cap
        or len(after[seed]['output_ids']) != cap
        or before[seed]['output_ids'][:cap] != after[seed]['output_ids']]
    bad_state = [seed for seed in expected if seed not in original or seed not in actual
        or original[seed] != actual[seed]]
    result['frozen_output'] = dict(reference=str(reference), expected=len(expected),
        failures=bad_output, passed=not bad_output,
        scope=f'Every one of the {cap} requested output tokens must equal the same prefix of the frozen Native 16-token output. This is a separate short-output discovery workload, not the primary 2048/16 workload.')
    result['frozen_full_state'] = dict(reference=str(reference), expected=len(expected),
        failures=bad_state, passed=not bad_state,
        scope='All original valid prompt KV-plane and hidden-state hashes must match. No prompt token or recurrent Loop is omitted.')
    result['passed'] &= not bad_output and not bad_state
    write(folder/'qualification_with_frozen_reference.json', result)
    return result
