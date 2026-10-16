"""Check frozen encoding and completed-cap output decoding without GPU use."""
import argparse
import sys
parser=argparse.ArgumentParser()
parser.add_argument('--version',choices=['4.54.1','4.57.1'],required=True)
parser.add_argument('--full-output',action='store_true')
options=parser.parse_args()
if options.version=='4.54.1':
    sys.path.insert(0,'/home/esjung/external/ouro-reasoning-fidelity-assets/hf454_deps')
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer


def main():
    import transformers,tokenizers
    from transformers import AutoTokenizer
    assert transformers.__version__==options.version
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    card=read(ROOT/'results/model_tokenizer_audit.json')['card_example_rendering']
    direct={}
    messages=[dict(role='user',content='Solve: If 2x + 3 = 11, what is x?')]
    for arm in ['P0','P1']:
        kwargs=dict(tokenize=True,add_generation_prompt=True,return_tensors='pt')
        if arm=='P1':
            kwargs['enable_thinking']=True
        actual=tokenizer.apply_chat_template(messages,**kwargs).tolist()[0]
        direct[arm]=dict(matches_frozen_card_ids=actual==card[arm]['ids'],IDs=actual)
    input_mismatches=[]
    frozen=read(ROOT/'results/frozen_questions.json')
    for row in frozen:
        for arm in ['P0','P1']:
            text=row['arms'][arm]['text']
            ids=tokenizer.encode(text,add_special_tokens=False)
            old=row['arms'][arm]['ids']
            if ids!=old:
                input_mismatches.append(dict(id=row['id'],dataset=row['dataset'],arm=arm,
                    frozen_length=len(old),new_length=len(ids),
                    first_divergence=next((i for i,(a,b) in enumerate(zip(old,ids)) if a!=b),None)))
    root=ROOT/'results/quality_v1'
    if options.full_output:
        assert read(ROOT/'results/quality_collection_and_scoring_complete.json')['success']
        paths=sorted(root.glob('*_cap*_wave*.json'))
        rows=[r for path in paths for r in read(path)]
    else:
        assert read(root/'aime24_sampling_cap8192_summary.json')['collection_complete']
        rows=[r for path in sorted(root.glob('aime24_sampling_cap8192_wave*.json')) for r in read(path)]
        assert len(rows)==90
    quality_count=len(rows)
    for path in (ROOT/'results/native_fidelity_TP4_v1').rglob('*cap8192_repeat0.json'):
        row=read(path)
        if row.get('id'):
            rows.append(dict(problem_id=row['id'],generated_ids=row['ids'],text=row['text'],
                             source='Native TP4 fidelity continuity, not full quality.'))
    differences=[]
    answer_differences=[]
    for row in rows:
        decoded=tokenizer.decode(row['generated_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)
        item=dict(id=row['problem_id'],seed=row.get('seed'),
                  source=row.get('source',f"Frozen quality {row.get('dataset')} cap{row.get('cap')}"))
        if decoded!=row['text']:
            differences.append(item | dict(native_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                reference_decode_sha256=hashlib.sha256(decoded.encode()).hexdigest(),
                first_text_divergence=next((i for i,(a,b) in enumerate(zip(decoded,row['text'])) if a!=b),None)))
        if final_answer(decoded)!=final_answer(row['text']):
            answer_differences.append(item)
    name='output_text_full_audit' if options.full_output else 'output_text_audit'
    target=ROOT/f'results/{name}_{options.version}.json'
    assert not target.exists()
    write(target,dict(transformers=transformers.__version__,tokenizers=tokenizers.__version__,
        tokenizer_config_sha256=sha(MODEL/'tokenizer_config.json'),
        direct_card_chat_template_API=direct,
        input_arms_checked=len(frozen)*2,input_encoding_mismatches=input_mismatches,
        quality_output_sequences_checked=quality_count,
        output_sequences_checked=len(rows),output_text_differences=differences,
        final_answer_extraction_differences=answer_differences,
        primary_scores_changed=False,GPU_used=False,
        scope='CPU tokenizer audit only. It does not validate model computation or the private author evaluator.'))
    print(dict(version=options.version,input_mismatches=len(input_mismatches),outputs=len(rows),
        output_text_differences=len(differences),answer_extraction_differences=len(answer_differences)))


if __name__=='__main__':
    main()
