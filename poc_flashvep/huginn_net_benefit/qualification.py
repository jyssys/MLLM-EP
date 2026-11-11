"""Extend the frozen prompt contract with explicit final Decode state checks."""
from .common import ROOT, json, write


def final_decode(name, reference=None):
    def load(run):
        records = [row for path in (ROOT / 'raw' / run).glob('post_decode_state_*.json')
            for row in json.loads(path.read_text())['rows']]
        by_seed = {r['seed']: r for r in records}
        assert len(by_seed) == len(records), 'Duplicate final Decode state'
        return by_seed
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    observed = load(name)
    expected = set(range(cfg['seed'], cfg['seed']+cfg['count']))
    failures = []
    if set(observed) != expected:
        failures.append('FINAL_STATE_COVERAGE')
    if reference:
        native = load(reference)
        for seed in expected:
            if seed not in observed or seed not in native or observed[seed] != native[seed]:
                failures.append(seed)
    result = dict(name=name, reference=reference, expected=cfg['count'], observed=len(observed),
        failures=failures, passed=not failures,
        scope='Every semantically valid final Decode KV plane and final hidden state. Reference runs establish Native states. This is a synchronized diagnostic, not a serving metric.')
    write(ROOT / 'raw' / name / 'post_decode_qualification.json', result)
    return result
