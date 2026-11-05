"""Preserve the first model and validate the corrected calibration prospectively."""
import argparse
from .common import *
from .prepare import configuration
from .oracle import fit_and_freeze, predict_configuration, headroom

MODEL = 'oracle_cost_model_frozen_v2.json'
PREDICTIONS = 'predictions_v2'


def prepare():
    initialize()
    original = ROOT / 'oracle_cost_model_frozen.json'
    model = fit_and_freeze(MODEL, PREDICTIONS)
    selected = json.loads((ROOT / 'state/frozen_static_selection.json').read_text())['selected']
    ledger = dict(status='PROSPECTIVE_VERSION2_PREPARATION', version1_SHA256=digest(original),
        version2_SHA256=digest(ROOT / MODEL), results=[], supplementary_configs=[],
        reason='The first model used CUDA stream entry minus host submission for preparation. This can double-count an earlier queued GPU batch. Version2 uses host ModelRunner._execute entry instead.',
        protocol_deviation='Version1 validation dispatches already began. Preserve every original model and prediction. Exclude pre-existing dispatches from Version2. Fit Version2 from calibration only before its future validation predictions.',
        thresholds_unchanged=True, fitting_uses_heldout_outcomes=False)
    methods = {m: list(dict.fromkeys(['native', 'zero'] + [r['method'] for r in choices.values()]))
               for m, choices in selected.items()}
    for start in (1, 2, 3):
        for m in ('huginn', 'ouro'):
            for markers in (False, True):
                label = 'on' if markers else 'off'
                for method in methods[m]:
                    cfg = configuration(m, method, 128, f'd_{m}_{method}_heldout_s{start}_{label}128',
                        split='heldout', start=start, markers=markers, arrival_seed=2026110600 + start)
                    result = predict_configuration(cfg, MODEL, PREDICTIONS)
                    ledger['results'].append(dict(name=cfg['name'], status=result['status']))
    # These controls can supply missing prospective coverage. They are not
    # added to the main serving ratios, and they do not retune the model.
    for m in ('huginn', 'ouro'):
        for method in methods[m]:
            first = ROOT / PREDICTIONS / f'd_{m}_{method}_heldout_s1_off128.json'
            if method == 'zero' or 'metrics' in json.loads(first.read_text()):
                continue
            for markers in (False, True):
                label = 'on' if markers else 'off'
                cfg = configuration(m, method, 128, f'v_{m}_{method}_heldout_s4_{label}128',
                    split='heldout', start=4, markers=markers, arrival_seed=2026110604)
                cfg.update(oracle_validation_only=True, phase='ORACLE_VERSION2_PROSPECTIVE_VALIDATION_ONLY')
                result = predict_configuration(cfg, MODEL, PREDICTIONS)
                ledger['supplementary_configs'].append(cfg['name'])
    ledger['status'] = 'PROSPECTIVE_VERSION2_PREDICTIONS_PREPARED'
    write(ROOT / 'state/oracle_version2_protocol.json', ledger)
    print(json.dumps(dict(status=ledger['status'], predicted=len(ledger['results']),
        excluded=sum(r['status'].startswith('EXCLUDED') for r in ledger['results']),
        supplementary_configs=ledger['supplementary_configs'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'validate'))
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    else:
        headroom(MODEL, PREDICTIONS)
