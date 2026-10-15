"""Freeze real source text and model-specific token IDs before GPU results."""
import datetime
import gzip
import random
import sys
from poc_looped_multigpu_kv_placement.contract import *


def main():
    target=ROOT/'results/frozen_workload.json';assert not target.exists()
    sys.path.append('/home/esjung/external/multibd-poc-deps')
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    previous=json.loads((PROJECT/'poc_dllm_prefix_canvas_cp/results/frozen_manifest.json').read_text())
    texts={};sources={};rng=random.Random(41015)
    for domain in ['math','document']:
        source=previous['sources'][domain]
        path=hf_hub_download(source['repo'],source['file'],repo_type='dataset',revision=source['revision'])
        assert sha(path)==source['file_sha256']
        data=pq.read_table(path).to_pydict()
        values=(['Question: '+q+'\nSolution: '+a for q,a in zip(data['question'],data['answer'])]
            if domain=='math' else [x for x in data['text'] if len(x)>100])
        rng.shuffle(values);texts[domain]='\n\n'.join(values);sources[domain]=source
    pieces=[]
    for source in previous['sources']['coding']['files']:
        assert sha(source['path'])==source['sha256']
        pieces.append('# Source: '+str(Path(source['path']).name)+'\n'+Path(source['path']).read_text())
    texts['coding']='\n\n'.join(pieces);sources['coding']=previous['sources']['coding']
    counts={'coding':3,'math':3,'document':2}
    tasks={
        'coding':'Write a Python function that counts duplicate list items. Return code only.',
        'math':'A shop sells 37 notebooks at 6 dollars each and refunds 18 dollars. Compute the net revenue.',
        'document':'Summarize the factual content of the preceding reference excerpt. Do not invent facts.'}
    rows=[];manifest=[]
    for model in MODELS:
        tokenizer=AutoTokenizer.from_pretrained(model_path(model),trust_remote_code=True)
        prefixes=[1024,4096,8192,16384,32704] if model=='ouro' else [1024,4032]
        for domain,count in counts.items():
            encoded=tokenizer.encode(texts[domain],add_special_tokens=False)
            assert len(encoded)>=count*max(prefixes),(model,domain,len(encoded))
            for index in range(count):
                head=tokenizer.encode('Reference excerpt:\n',add_special_tokens=False)
                tail=tokenizer.encode('\n\nTask: '+tasks[domain]+'\nResponse:\n',add_special_tokens=False)
                background=encoded[index*max(prefixes):(index+1)*max(prefixes)]
                rid=f'{domain}_{index:02d}'
                requests={str(p):head+background[:p-len(head)-len(tail)]+tail for p in prefixes}
                rows.append(dict(model=model,request_id=rid,domain=domain,prefix_token_ids=requests))
                manifest.append(dict(model=model,request_id=rid,domain=domain,prefixes=prefixes,
                    source_token_offset=index*max(prefixes),task=tasks[domain],
                    prompt_text_sha256={p:hashlib.sha256(tokenizer.decode(ids).encode()).hexdigest() for p,ids in requests.items()},
                    prefix_ID_sha256={p:hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest() for p,ids in requests.items()}))
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','wt') as stream:json.dump(rows,stream)
    write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),requests=manifest,sources=sources,
        selection='Metadata-only contiguous, nonoverlapping source-token ranges within each model/domain. No output or performance selection.',
        source_caveat='Token ranges differ across tokenizers. Article/file disjointness is not claimed.',
        calibration=8,heldout='NOT_SELECTED_OR_EXECUTED. Freeze a fresh confirmation cohort only if discovery gate passes.',
        output_tokens=64,temperature=0,repeats=3,warmup=3,seed=41015,
        ouro_depth=4,huginn_depth=16,huginn_cache='full',
        context_rule='Ouro32704+64 stays within32K SFT. Huginn4032+64 stays within4096 native block_size.',
        quality='Bounded performance probes are not benchmark accuracy. Natural EOS probes are separate.',
        token_artifact_sha256=sha(ROOT/'results/frozen_tokens.json.gz')))
    print('Frozen8 calibration requests per model before GPU results.')


if __name__=='__main__':main()
