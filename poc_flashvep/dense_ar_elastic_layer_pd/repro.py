"""Compare original greedy token sequences without weakening the symptom."""
import argparse
from .common import *
from .analyze import records


def archived():
    reference={r['sample_id']:r for r in records(PREVIOUS/'raw/coarse_O16_N64_q4.777_native16_s1_v1/requests.jsonl')}
    candidate=records(PREVIOUS/'raw/HND_lean_chunk512_O16_N64_v1/requests.jsonl')
    failures=[]
    for r in candidate:
        expected=reference[r['sample_id']]
        assert r['input_ids']==expected['input_ids'] and r['seed']==expected['seed']
        if r['output_ids']==expected['output_ids']:continue
        first=next(i for i,(a,b) in enumerate(zip(expected['output_ids'],r['output_ids'])) if a!=b)
        failures.append(dict(sample_id=r['sample_id'],prompt_id=r['prompt_id'],first_different_output_index=first,
            expected=expected['output_ids'][first],actual=r['output_ids'][first]))
    value=dict(candidate='HND_lean_chunk512_O16_N64_v1',reference='coarse_O16_N64_q4.777_native16_s1_v1',
        compared=len(candidate),mismatches=failures,passed=len(candidate)==64 and not failures,
        evidence='TRACE_ANALYSIS',scope='Replay of original input/seed/token records. New GPU runs must qualify a fix independently.')
    write(ROOT/'analysis/archived_chunk512_red_repro.json',value)
    print(json.dumps(value,indent=2),flush=True)
    assert value['passed'],'ORIGINAL_GREEDY_OUTPUT_MISMATCH'


def main():
    initialize();p=argparse.ArgumentParser();p.add_argument('--archived',action='store_true');p.parse_args();archived()


if __name__=='__main__':main()
