"""Version the prompt recipe before inference. Never replace the first manifest."""
import hashlib
import sys
from poc_decode_first_depth_aware_dp.contract import *


def main():
    sys.path.insert(0,str(CDB/'src'))
    from looped_cdb.eval.gsm8k import _FEWSHOT_EXAMPLES
    source=ROOT/'artifacts/frozen_workload.json';output=ROOT/'artifacts/quality_protocol_v2.json'
    assert not output.exists()
    first=read(source)
    demos='\n\n'.join('Q: '+r['question']+'\nA: '+r['target'] for r in _FEWSHOT_EXAMPLES[:3])
    rows=[]
    for row in first['rows']:
        prompt=(demos+'\n\nQ: '+row['question']+'\nA:' if row['dataset']=='gsm8k' else
            'Problem: '+row['question']+'\nSolution: Let us reason step by step. Put the final answer in \\boxed{...}.')
        rows.append(dict(row,prompt=prompt,stop_strings=['Q:','</s>','<|im_end|>'] if row['dataset']=='gsm8k' else []))
    write(output,dict(rows=rows,parent_manifest_sha256=sha(source),before_any_inference=True,
        reason='Use the official CDB first-three canonical CoT demonstrations instead of the initial short-answer training examples.',
        prompt_style='GSM8K official CDB 3-shot CoT. MATH-500 zero-shot boxed answer. No base-model chat template.',
        source=str(CDB/'src/looped_cdb/eval/gsm8k.py'),source_sha256=sha(CDB/'src/looped_cdb/eval/gsm8k.py'),
        prompt_sha256={r['id']:hashlib.sha256(r['prompt'].encode()).hexdigest() for r in rows},
        max_new_tokens=4096,followup_cap=8192,pilot_per_domain=16,temperature=0,
        scoring='Same explicit answer parser for fixed and dynamic, GSM numeric and MATH MathVerify. No candidate-only extraction fallback.'))
    print('QUALITY_RECIPE_V2_FROZEN',len(rows),'rows')


if __name__=='__main__':main()
