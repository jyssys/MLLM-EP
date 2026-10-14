"""Freeze real-text context excerpts and complete tasks before profiling."""
import datetime
import gzip
import random
import sys
import sysconfig
import numpy as np
from huggingface_hub import HfApi,hf_hub_download
from transformers import AutoTokenizer
from poc_dllm_prefix_canvas_cp.contract import *


def main():
    target=ROOT/'results/frozen_manifest.json';assert not target.exists()
    sys.path.append('/home/esjung/external/multibd-poc-deps')
    import pyarrow.parquet as pq
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    rng=random.Random(19411)
    api=HfApi();sources={};corpora={}
    for domain,repo,file,revision in [
        ('document','Salesforce/wikitext','wikitext-2-raw-v1/test-00000-of-00001.parquet','b08601e04326c79dfdd32d625aee71d232d685c3'),
        ('math','openai/gsm8k','main/train-00000-of-00001.parquet',None)]:
        revision=revision or api.dataset_info(repo).sha
        local=hf_hub_download(repo,file,repo_type='dataset',revision=revision)
        data=pq.read_table(local).to_pydict()
        if domain=='math':texts=['Question: '+q+'\nSolution: '+a+'\n' for q,a in zip(data['question'],data['answer'])]
        else:texts=[t for t in data['text'] if len(t)>100]
        rng.shuffle(texts)
        corpora[domain]=tokenizer.encode('\n\n'.join(texts),add_special_tokens=False)
        sources[domain]=dict(repo=repo,revision=revision,file=file,file_sha256=sha(local),tokens=len(corpora[domain]))
    stdlib=Path(sysconfig.get_paths()['stdlib'])
    files=[p for p in stdlib.rglob('*.py') if not any(s in p.parts for s in ['site-packages','test','tests','__pycache__'])]
    files=sorted(files);rng.shuffle(files)
    tokens=[];code_sources=[]
    for path in files:
        text=path.read_text(errors='replace')
        tokens+=tokenizer.encode('# Source: '+str(path.relative_to(stdlib))+'\n'+text+'\n\n',add_special_tokens=False)
        code_sources.append(dict(path=str(path),sha256=sha(path)))
        if len(tokens)>240000:break
    corpora['coding']=tokens
    sources['coding']=dict(source='CPython standard library, PSF-licensed source excerpts',
        python=sys.version,files=code_sources,tokens=len(tokens))
    counts={'coding':12,'math':12,'document':8};cal={'coding':3,'math':3,'document':2}
    tasks={
        'coding':'The source excerpt above is read-only context. Write a short Python function that counts duplicate items in a list. Return code only.',
        'math':'The worked examples above are context. A shop sells 37 notebooks at 6 dollars each and refunds 18 dollars. Compute the net revenue and explain the calculation.',
        'document':'Write a short factual summary of the document excerpt above. Do not add facts absent from the excerpt.'}
    rows=[];blocks=[];manifest=[]
    for domain,count in counts.items():
        assert len(corpora[domain])>=count*16384
        for index in range(count):
            rid=f'{domain}_{index:02d}';split='calibration' if index<cal[domain] else 'held_out'
            context=corpora[domain][index*16384:(index+1)*16384]
            head=tokenizer.encode('<role>SYSTEM</role>detailed thinking off<|role_end|><role>HUMAN</role>Reference excerpt:\n',add_special_tokens=False)
            tail=tokenizer.encode('\n\nTask: '+tasks[domain]+'<|role_end|><role>ASSISTANT</role>',add_special_tokens=False)
            row=dict(request_id=rid,domain=domain,split=split,head=head,tail=tail,background=context)
            rows.append(row)
            hashes={str(p):__import__('hashlib').sha256(np.asarray(head+context[:p-len(head)-len(tail)]+tail,dtype=np.int32).tobytes()).hexdigest() for p in PREFIXES}
            manifest.append(dict(request_id=rid,domain=domain,split=split,task=tasks[domain],
                corpus_offset=index*16384,head_tokens=len(head),tail_tokens=len(tail),prefix_ID_sha256=hashes))
    rng.shuffle(rows)
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','wt') as out:json.dump(rows,out)
    write(target,dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=19411,
        calibration=8,held_out=24,selection='metadata-only, disjoint background token ranges within each public domain corpus; no output/routing selection',
        task_selection='one frozen task/domain; background excerpt changes by request and prefix length; not an accuracy benchmark',
        source_caveat='corpus ranges are disjoint; not guaranteed article/file/benchmark-question disjoint',
        prefixes=PREFIXES,output_budget=64,block=32,batch=1,threshold=.9,temperature=0,
        prefilling_limit=MAX_CONTEXT,runtime_context=MAX_CONTEXT,sources=sources,requests=manifest,
        token_artifact_sha256=sha(ROOT/'results/frozen_tokens.json.gz'),
        phases=dict(initial='native block-causal prefill',warm='same-request post-prefill refinement, not cross-request prefix caching'),
        repeats=dict(warmup=3,clean=3,coarse_event=2,heavy_trace='one request per prefix, separate',profiler='one bounded sample if needed'),
        no_candidate_before_stage_A=True,held_out_execution='only after Stage A GO',
        stopping=dict(attention_target_share=.25,attainable_cold_gain=.10,observer_CI_upper=.10)))
    print('FROZEN 8 calibration +24 untouched held-out; prefixes',PREFIXES)


if __name__=='__main__':main()
