"""Explain one official scorer difference without changing scores or answers."""
import json
import time

import regex

from .common import ROOT, digest, preserve, write
from .quality_audit import source_definitions


def main():
    preserve()
    path = ROOT / 'analysis/C14_Agg_EPD2_N3_quality_v0.json'
    audit = json.loads(path.read_text())
    namespace = dict(re=regex, List=list)
    provenance = source_definitions('Mantis_mantis_benchmark_eval.py.json',
        ['parse_answer', 'get_option', 'get_prediction'], namespace)
    request_id = 'TIGER-Lab__Mantis-Eval__dj_6'
    names = ['reference_main400_natural_template_v2_v0',
             'base_C14_e_pd_S2_natural100_q2_v0']
    rows = []
    for name in names:
        run = audit['runs'][name]
        item, = [r for r in run['requests'] if r['request_id'] == request_id]
        raw = item['raw_answer']
        parsed = namespace['parse_answer'](raw)
        rows.append(dict(run=name, request_id=request_id,
            raw_answer=raw, official_parse_answer=parsed,
            official_first_option=namespace['get_option'](parsed),
            official_score=item['score'], official_parsed_answer=item['parsed_answer'],
            reference_answers=item['reference_answers'],
            source_mean_score=run['source_results']['TIGER-Lab/Mantis-Eval']['mean_score'],
            literal_final_line=raw.strip().splitlines()[-1]))
    assert rows[0]['official_score'] == 1 and rows[1]['official_score'] == 0
    assert rows[0]['literal_final_line'] == rows[1]['literal_final_line'] == '(A) The first image'
    assert rows[0]['official_first_option'] == 'A' and rows[1]['official_first_option'] == 'T'
    result = dict(time=time.time(), evidence='DIAGNOSTIC', input_SHA256=digest(path),
        scorer_provenance=provenance, rows=rows,
        conclusion='Both unchanged responses end with the same literal option A. '
            'The pinned parser recognizes "the answer is" but not "the correct answer is". '
            'The second response therefore retains its opening narrative and yields option T. '
            'The official score remains 0. No answer, official scorer, or quality threshold changes.',
        scope='One request and one scoring discrepancy. This is not a replacement task metric '
            'or a full quality PASS. MMLongBench-Doc official extraction remains NOT_RUN.')
    output = ROOT / 'analysis/C14_Mantis_parser_diagnostic_v0.json'
    assert not output.exists()
    write(output, result)
    preserve()
    print('UNCHANGED_OFFICIAL_SCORES', [(r['official_score'], r['official_first_option']) for r in rows])


if __name__ == '__main__':
    main()
