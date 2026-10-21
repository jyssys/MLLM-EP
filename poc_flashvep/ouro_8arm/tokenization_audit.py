"""Audit cross-version input differences. The alternative IDs never enter quality runs."""
import collections
import transformers
from poc_flashvep.ouro_request_pp.model_adapter import tokenizer
from .contract import *


def main():
    tok=tokenizer();mismatches=[];counts=collections.Counter();stop_strings=set()
    for row in dataset(False):
        other=tok.encode(row['prompt'],add_special_tokens=False);gold=row['input_ids']
        stop_strings.update(row['stop_strings'])
        if other!=gold:
            counts[row['dataset']]+=1
            index=next((i for i,(a,b) in enumerate(zip(other,gold)) if a!=b),min(len(other),len(gold)))
            mismatches.append(dict(id=row['id'],pilot=row['pilot'],canonical_length=len(gold),alternative_length=len(other),first_difference=index,
                canonical_slice=gold[max(0,index-3):index+4],alternative_slice=other[max(0,index-3):index+4]))
    write(ROOT/'tokenization_audit.json',dict(canonical_transformers=read(ROOT/'frozen_prompt_ids.json')['transformers'],
        alternative_transformers=transformers.__version__,mismatch_counts=counts,mismatches=mismatches,
        alternative_stop_IDs={s:tok.encode(s,add_special_tokens=False) for s in sorted(stop_strings)},
        rule='Every current quality runtime consumes canonical explicit IDs. Alternative tokenization is diagnostic only.'))
    print('TOKENIZATION_AUDIT',dict(counts),len(mismatches),flush=True)


if __name__=='__main__':main()
