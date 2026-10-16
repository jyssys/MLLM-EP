"""Repeat the exact first full-quality request history in one fresh engine."""
import argparse
import os
import time
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--core-multiprocessing',choices=['0','1'],default='1')
    args=parser.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4,5,6,7'
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']=args.core_multiprocessing
    from vllm import LLM,SamplingParams
    output=Path(os.environ['LOOPED_KV_OUTPUT'])
    kwargs=read(ROOT/'results/quality_v1/engine_contract.json')
    write(output/'engine_contract.json',kwargs)
    llm=LLM(**kwargs)
    owners=llm.collective_rpc('inspect_ownership',timeout=120)
    assert len(owners)==4 and all(r['cache_slots']==192 for r in owners)
    assert [r['GPU_uuid'].removeprefix('GPU-') for r in owners]==UUIDS
    write(output/'ownership.json',owners)
    questions=read(ROOT/'results/frozen_questions.json')[:4]
    records=[]
    for cap in [256,256,8192,8192]:
        params=[SamplingParams(temperature=1.,top_p=.7,top_k=-1,seed=41015,
            max_tokens=cap,ignore_eos=False,repetition_penalty=1.,stop=[]) for row in questions]
        start=time.perf_counter()
        values=llm.generate([dict(prompt_token_ids=r['arms']['P1']['ids']) for r in questions],params,use_tqdm=False)
        records.append(dict(cap=cap,wall_ms=(time.perf_counter()-start)*1000,
            outputs=[dict(id=r['id'],generated_ids=list(v.outputs[0].token_ids),text=v.outputs[0].text,
                          finish_reason=v.outputs[0].finish_reason) for r,v in zip(questions,values)]))
        write(output/'request_history.json',records)
    reference=read(ROOT/'results/quality_v1/aime24_sampling_cap8192_wave00.json')
    comparisons=[]
    for a,b in zip(reference,records[2]['outputs']):
        x,y=a['generated_ids'],b['generated_ids']
        comparisons.append(dict(id=a['problem_id'],equal=x==y,
            first_divergence=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None),
            reference_tokens=len(x),new_tokens=len(y)))
    history=[]
    for a,b in zip(records[2]['outputs'],records[3]['outputs']):
        x,y=a['generated_ids'],b['generated_ids']
        history.append(dict(id=a['id'],equal=x==y,
            first_divergence=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),None),
            first_tokens=len(x),second_tokens=len(y)))
    write(output/'restart_history_comparison.json',dict(success=True,core_multiprocessing=args.core_multiprocessing,
        identical_client_request_history=True,comparisons=comparisons,all_equal=all(r['equal'] for r in comparisons),
        within_session_after_one_long_wave=history,
        within_session_long_waves_equal=all(r['equal'] for r in history),
        scope='No sampler-state or first-differing-logit trace. This comparison does not establish the exact physical cause.'))
    os._exit(0)


if __name__=='__main__':
    main()
