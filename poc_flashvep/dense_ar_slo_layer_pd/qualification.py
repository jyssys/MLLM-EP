"""Check output parity against the immutable full Native reference."""
from .common import *
from .analyze import records


def against_previous(name):
    old=records(PREVIOUS/'raw/baseline_Native_O200_N64_q4p8_s21_v1/requests.jsonl')
    reference={r['prompt_id']:r for r in old}
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    new=records(ROOT/'raw'/name/'requests.jsonl');failures=[]
    assert cfg.get('ignore_eos',True),'Natural EOS needs a separate reference'
    for row in new:
        r=reference.get(row['prompt_id'])
        if r is None or row['input_ids']!=r['input_ids']:
            failures.append(dict(sample_id=row['sample_id'],reason='REFERENCE_INPUT_MISSING'));continue
        if row['output_ids']!=r['output_ids'][:len(row['output_ids'])]:
            failures.append(dict(sample_id=row['sample_id'],reason='EXACT_GREEDY_OUTPUT_MISMATCH'))
    result=dict(name=name,compared=len(new),expected=cfg['count'],failures=failures,
        passed=len(new)==cfg['count'] and not failures,evidence='CORRECTNESS_ONLY',
        scope='Exact tokens at identical real prompt IDs and token arrays. The frozen Native reference supplies up to200 tokens. This check does not establish KV/Hidden state parity or sustainable performance.')
    write(ROOT/'analysis'/(name+'_output_gate.json'),result);return result


def against_unseen(name):
    gate=json.loads((ROOT/'analysis/unseen_reference_correctness.json').read_text())
    assert gate['passed'],'The independent unseen reference must pass first'
    reference={r['prompt_id']:r for r in records(ROOT/'raw'/gate['reference']/'requests.jsonl')}
    cfg=json.loads((ROOT/'configs'/(name+'.json')).read_text())
    assert cfg.get('ignore_eos',True),'Natural EOS requires a separate reference'
    rows=records(ROOT/'raw'/name/'requests.jsonl');failures=[]
    for row in rows:
        old=reference.get(row['prompt_id'])
        if old is None or old['input_ids']!=row['input_ids']:
            failures.append(dict(sample_id=row['sample_id'],reason='UNSEEN_REFERENCE_INPUT_MISSING'))
        elif row['output_ids']!=old['output_ids'][:len(row['output_ids'])]:
            failures.append(dict(sample_id=row['sample_id'],reason='EXACT_UNSEEN_OUTPUT_MISMATCH'))
    result=dict(name=name,reference=gate['reference'],compared=len(rows),expected=cfg['count'],
        failures=failures,passed=len(rows)==cfg['count'] and not failures,evidence='CORRECTNESS_ONLY',
        scope='Exact greedy tokens against the independently qualified unseen real-text reference. Reference latency is not an evaluation result. Prompt cycles do not add independent server starts.')
    write(ROOT/'analysis'/(name+'_output_gate.json'),result);return result
