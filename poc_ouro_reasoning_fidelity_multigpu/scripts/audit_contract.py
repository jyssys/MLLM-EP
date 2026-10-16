"""CPU-only provenance and parser checks. Do not score generated answers here."""
import importlib.metadata
import subprocess
from poc_ouro_reasoning_fidelity_multigpu.contract import *
from poc_looped_exact_depth_chunk_pipeline.scoring import final_answer, aime_integer

def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    config=read(MODEL/'config.json')
    template=tokenizer.chat_template
    cards={}
    for arm,thinking in [('P0',False),('P1',True)]:
        text=tokenizer.apply_chat_template([dict(role='user',content='Solve: If 2x + 3 = 11, what is x?')],
            tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
        ids=tokenizer.encode(text,add_special_tokens=False)
        cards[arm]=dict(text=text,escaped_repr=repr(text),ids=ids,last32_ids=ids[-32:])
    tests=[('The result is \\boxed{007}.',7),('\\boxed{3} then \\boxed{12}',12),
        ('Final answer: 42',42),('Reasoning contains 42 but no final answer.',None),
        ('\\boxed{1000}',None),('\\boxed{-1}',None),('\\boxed{\\text{7}}',7)]
    outcomes=[]
    for text,expected in tests:
        answer,rule=final_answer(text)
        actual=aime_integer(answer)
        outcomes.append(dict(input=text,expected=expected,actual=actual,rule=rule,PASS=actual==expected))
    assert all(r['PASS'] for r in outcomes)
    weights=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in MODEL.glob('*.safetensors')]
    tokens={name:dict(token=getattr(tokenizer,name+'_token'),id=getattr(tokenizer,name+'_token_id')) for name in ['bos','eos','pad']}
    write(ROOT/'results/model_tokenizer_audit.json',dict(config=config,special_tokens=tokens,
        thinking_id=tokenizer.convert_tokens_to_ids('<think>'),end_thinking_id=tokenizer.convert_tokens_to_ids('</think>'),
        eos_decode_keep=tokenizer.decode([2],skip_special_tokens=False),eos_decode_skip=tokenizer.decode([2],skip_special_tokens=True),
        card_example_rendering=cards,card_example_vs_primary='Different user content. P2 is a formatting audit, not another tuned evaluation arm.',
        template=template,template_sha256=hashlib.sha256(template.encode()).hexdigest(),weights=weights,
        generation_config=read(MODEL/'generation_config.json') if (MODEL/'generation_config.json').exists() else None,
        cpu_versions={p:importlib.metadata.version(p) for p in ['torch','transformers','tokenizers']},
        native_versions=subprocess.check_output([PYTHON,'-c','import torch,transformers,vllm; print(torch.__version__, transformers.__version__, vllm.__version__, torch.version.cuda)'],text=True)))
    write(ROOT/'results/grader_unit_tests.json',dict(tests=outcomes,PASS=True,
        wrong_answer_controls=[dict(reference=7,prediction=8,correct=False),dict(reference=7,prediction=None,correct=False)],
        capped_rule='A matching extracted answer is not completed accuracy if finish_reason is length.'))
    print('Audit and parser unit tests saved',flush=True)

if __name__=='__main__':
    main()
