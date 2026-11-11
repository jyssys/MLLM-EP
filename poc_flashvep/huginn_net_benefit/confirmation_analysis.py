"""Report held-out run-level uncertainty and retain all Zero sensitivity tests."""
import argparse
from .common import *
from .serving_exactness import compare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='confirm')
    args = parser.parse_args()
    initialize()
    design = json.loads((ROOT/'state'/(args.tag+'_confirmation_design.json')).read_text())
    replacement_path = ROOT/'state'/(args.tag+'_isolation_replacements.json')
    replacements = json.loads(replacement_path.read_text())['replacements'] if replacement_path.exists() else {}
    from .confirmation_isolation import resolve
    rows, exactness = [], []
    for planned in design['jobs']:
        name = resolve(planned, replacements)
        cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
        value = analysis().metrics(name)
        assert value['complete'] and value['count'] == 128, name
        label = 'zerorepeat' if '_zero_repeat_seed' in name else cfg['confirmation_label']
        rows.append(dict(value, actual_name=name,
            name=f'paired_huginn_{label}_seed{cfg["arrival_seed"]}', label=label))
        reference = f'{args.tag}_zero_seed{cfg["arrival_seed"]}'
        if name != reference:
            exactness.append(compare(reference, name))
    from ..completion_aware_loop_pd.serving_audit import paired, csv_write
    labels = list(design['arms'])
    ratios = paired(rows, arms=[*labels, 'zerorepeat'],
        references=['zero', 'zerorepeat', *[label for label in labels if label != 'zero']])
    target = ROOT/'confirmation'
    target.mkdir(exist_ok=True)
    write(target/(args.tag+'_serving_results.json'), rows)
    csv_write(target/(args.tag+'_serving_results.csv'), rows)
    write(target/(args.tag+'_paired_start_ratios.json'), ratios)
    csv_write(target/(args.tag+'_paired_start_ratios.csv'), ratios)
    write(target/(args.tag+'_output_exactness.json'), exactness)
    write(target/(args.tag+'_interpretation.json'), dict(
        independent_arrival_seeds=len(design['arrival_seeds']),
        process_starts=len(rows), all_outputs_exact=all(r['passed'] for r in exactness),
        inference_unit='An independent process start paired by Arrival seed. The repeated Zero is a sensitivity control. It is not an additional Arrival seed.',
        uncertainty='Start-level bootstrap. Three Arrival seeds give limited confidence coverage. No individual-token independence assumption applies.',
        throughput='Finite-cohort drain rates. Sustainable capacity is not established.',
        state_scope='Marker-free N128 output and full recurrent-work accounting. Intermediate-state equality comes from the required N8/N32 diagnostics.',
        service_TTFT='NOT_MEASURED in marker-free confirmation. No Host timestamp substitutes for actual first GPU execution.'))
    write(target/(args.tag+'_isolation_account.json'), dict(replacements=replacements,
        included_process_starts=len(rows),
        scope='A predeclared clean repeat replaces a possible source-search overlap. The original artifacts remain. This does not add an independent Arrival seed.'))
    sensitivity = []
    for original, replacement in replacements.items():
        for status, name in (('POSSIBLE_CPU_IO_OVERLAP', original), ('CLEAN_REPLACEMENT', replacement)):
            sensitivity.append(dict(analysis().metrics(name), isolation_status=status,
                replaces=original, scope='Retain both results. Only the declared replacement enters the primary paired analysis.'))
    write(target/(args.tag+'_isolation_sensitivity.json'), sensitivity)
    csv_write(target/(args.tag+'_isolation_sensitivity.csv'), sensitivity)
    print(json.dumps(dict(starts=len(rows), output_checks=len(exactness),
        all_outputs_exact=all(r['passed'] for r in exactness))))


if __name__ == '__main__':
    main()
