"""Separate byte identity, full-work, output/state parity, and lifetime gates."""
from .common import *
from .run import execute


def qualify(name, reference=None):
    initialize()
    folder = ROOT / 'raw' / name
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    cached = folder / 'qualification.json'
    if cached.exists():
        return json.loads(cached.read_text())
    result = dict(name=name, complete=execute(name), passed=False)
    if result['complete']:
        from ..looped_pd_atomic_prefill import evidence
        from ..fluidtoken_loop_split.state_comparison import compare
        from ..dynamic_loop_split_serving.lifetime_gate import check
        from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
        with quiet_analysis(name + '_qualification'):
            result['work'] = evidence.full_work(name)
            before = json.loads((folder / 'before_measured.json').read_text())
            after = json.loads((folder / 'after_measured.json').read_text())
            keys = [key for key in after if '-decode-' in key]
            assert len(keys) == 1
            observed = after[keys[0]]['issued_loop_rows'] - before[keys[0]]['issued_loop_rows']
            decode = cfg['count'] * (cfg['cap'] - 1) * cfg['depth']
            frontier = cfg.get('native_frontier')
            expected = (cfg['count'] * (cfg['depth'] - frontier[0]) * (cfg['length'] - frontier[1])
                if frontier else cfg['count'] * (cfg['depth'] - cfg['native_partial_cut']) * cfg['length']
                if cfg.get('native_partial_cut') is not None else 0)
            if cfg.get('native_request_cuts'):
                expected = sum((cfg['depth'] - cut) * cfg['length'] for cut in cfg['native_request_cuts'])
            result['actual_offload'] = dict(D_prompt_rows=observed - decode, expected_D_prompt_rows=expected,
                passed=observed - decode == expected,
                scope='Measured active D recurrent rows minus exact post-first-token Decode rows. Not FLOPs.')
            if cfg.get('native_partial_cut') is None and not cfg.get('feature_native_wire_v5'):
                result['transfer'] = evidence.transfer(name)
            else:
                expected = set(range(cfg['seed'], cfg['seed'] + cfg['count']))
                checks = [json.loads(line) for path in folder.glob('KV_checks_*.jsonl')
                          for line in path.read_text().splitlines()]
                checks = [row for row in checks if row['seed'] in expected]
                by_seed = {row['seed']: row for row in checks}
                result['transfer'] = dict(compared=len(by_seed), expected=len(expected),
                    raw_rows_including_startup=len(checks),
                    passed={row['seed'] for row in checks} == expected and all(row['passed'] for row in checks),
                    scope='Receiver fingerprint versus source. Last observation per seed follows startup. Raw startup checks remain retained. All transferred valid KV and hidden bytes.')
            result['lifetime'] = check(name)
            if reference:
                result['output'] = evidence.compare(reference, name)
                result['state'] = compare(reference, name)
            result['passed'] = (result['work']['passed'] and result['actual_offload']['passed'] and result['transfer']['passed']
                and result['lifetime']['passed'] and result.get('output', {'passed': True})['passed']
                and result.get('state', {'exact': True})['exact'])
    write(cached, result)
    return result


def main():
    initialize()
    ledger = dict(status='N8_IN_PROGRESS', models={})
    for model in ('ouro', 'huginn'):
        row = ledger['models'][model] = {}
        reference = f'a_{model}_native_v2_state8'
        for path in ('native', 'zero', 'partial'):
            name = f'a_{model}_{path}_v2_state8'
            row[path] = qualify(name, reference if path != 'native' else None)
            write(ROOT / 'state/qualification.json', ledger)
            if path == 'native' and not row[path]['passed']:
                break
    ledger['status'] = 'N8_FINISHED_REVIEW_REQUIRED'
    write(ROOT / 'state/qualification.json', ledger)


if __name__ == '__main__':
    main()
