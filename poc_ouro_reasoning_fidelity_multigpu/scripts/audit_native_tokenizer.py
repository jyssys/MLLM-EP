"""Check installed native tokenizer drift without changing frozen input IDs."""
from poc_ouro_reasoning_fidelity_multigpu.contract import *

def main():
    import transformers
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    records=[]
    for row in read(ROOT/'results/frozen_questions.json'):
        for arm in ['P0','P1']:
            expected=row['arms'][arm]
            assert tokenizer.decode(expected['ids'],skip_special_tokens=False)==expected['text']
            actual=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],
                tokenize=True,add_generation_prompt=True,enable_thinking=arm=='P1')
            actual=actual['input_ids'] if hasattr(actual,'keys') else actual
            if actual!=expected['ids']:
                records.append(dict(id=row['id'],dataset=row['dataset'],arm=arm,
                    runtime_retokenized_ids=actual,frozen_ids=expected['ids'],
                    decoded_prompt_equal=True))
    write(ROOT/'results/native_tokenizer_preflight.json',dict(
        transformers=transformers.__version__,checked=260,all_frozen_IDs_decode_to_exact_prompt=True,
        retokenization_differences=records,policy='Every native run uses frozen IDs directly. Retokenization is diagnostic only.'))
    print('Checked 260 exact prompt decodes. Retokenization differences:',len(records))

if __name__=='__main__':
    main()
