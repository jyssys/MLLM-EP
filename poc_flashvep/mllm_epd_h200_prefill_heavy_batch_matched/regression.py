"""Repeat the fixed Calibration32 N1/N8 cases after admission changes."""
import asyncio
import random

from .runtime import module, require

def native_checks(rows, teacher_url, native_url):
    compare = module('native_common_control').compare_capture
    capture = module('token_correctness_control').capture
    indices = module('natural_token_correctness_control').INDICES
    results = {}
    for phase, cohort in [('N1', [0]), ('N8', indices)]:
        phases = []
        for index in cohort:
            case = f'irteam_native_{phase}_{index}'
            row = rows[index]
            teacher = capture(row, case + '_teacher', teacher_url, [(18841, 'P')])
            observed = capture(row, case + '_epd', native_url, [(18841, 'P'), (18842, 'D')])
            record = compare(row, case + '_teacher', case + '_epd', teacher, observed)
            require(record['passed'], f'Native {phase} gate failed for request {row["request_id"]}')
            phases.append({'request_id': row['request_id'], 'passed': record['passed'],
                           'strict_greedy_tokens': record['strict_greedy_tokens']})
        results[phase] = phases
    return results


def token_checks(rows, teacher_url, token_url, proxy_name):
    capture = module('token_correctness_control').capture
    token = module('natural_token_correctness_control')
    prefix = module('token_split').select_prefix
    refs = {}
    for index in token.INDICES:
        row = rows[index]
        name = f'irteam_token10_{index}_teacher'
        capture(row, name, teacher_url, [(18841, 'P')], prefix(row['total_input_tokens'], .10))
        refs[row['request_id']] = name
    results = {}
    n8_rows = [rows[index] for index in token.INDICES]
    random.Random(2026112404).shuffle(n8_rows)
    for phase, values in [('N1', [rows[0]]), ('N8', n8_rows)]:
        record = asyncio.run(token.cohort('irteam_token10_' + phase, values, token_url, .10, refs, proxy_name))
        require(record['passed'], f'Token10 {phase} gate failed')
        results[phase] = {'passed': record['passed'], 'requests': len(values)}
    return results
