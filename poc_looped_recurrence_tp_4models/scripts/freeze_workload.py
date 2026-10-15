"""Freeze source-based shapes and per-model teacher tokens before GPU outcomes."""
import datetime
import gzip
import random
import sys
from poc_looped_recurrence_tp_4models.contract import *


def main():
    target=ROOT/'results/frozen_workload.json';assert not target.exists()
    sys.path.append('/home/esjung/external/multibd-poc-deps')
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    previous=json.loads((PROJECT/'poc_looped_multigpu_kv_placement/results/frozen_workload.json').read_text())
    with gzip.open(PROJECT/'poc_looped_multigpu_kv_placement/results/frozen_tokens.json.gz','rt') as stream:
        old=[r for r in json.load(stream) if r['model']=='ouro']
    texts={};rng=random.Random(41015)
    for domain in ['math','document']:
        source=previous['sources'][domain]
        path=hf_hub_download(source['repo'],source['file'],repo_type='dataset',revision=source['revision'])
        assert sha(path)==source['file_sha256']
        data=pq.read_table(path).to_pydict()
        values=(['Question: '+q+'\nSolution: '+a for q,a in zip(data['question'],data['answer'])]
            if domain=='math' else [x for x in data['text'] if len(x)>100])
        rng.shuffle(values);texts[domain]='\n\n'.join(values)
    pieces=[]
    for source in previous['sources']['coding']['files']:
        assert sha(source['path'])==source['sha256']
        pieces.append('# Source: '+Path(source['path']).name+'\n'+Path(source['path']).read_text())
    texts['coding']='\n\n'.join(pieces)
    rows=[];manifest=[]
    for model in MODELS:
        tokenizer=AutoTokenizer.from_pretrained(model_path(model),trust_remote_code=True)
        by_domain={domain:tokenizer.encode(text,add_special_tokens=False) for domain,text in texts.items()}
        for row in old:
            domain=row['domain'];index=int(row['request_id'].split('_')[-1]);request=row['request_id']
            if model=='ouro':prompts={str(p):row['prefix_token_ids'][str(p)] for p in PREFIXES}
            else:
                source=by_domain[domain];start=index*33000;assert len(source)>=start+33000
                prompts={str(p):source[start:start+p] for p in PREFIXES}
            # A fixed source continuation is a hardware control, not a natural answer.
            teacher=by_domain[domain][index*33000+32768:index*33000+32768+64]
            assert len(teacher)==64
            rows.append(dict(model=model,request_id=request,domain=domain,prompt_ids=prompts,teacher_ids=teacher))
            manifest.append(dict(model=model,request_id=request,domain=domain,prompt_hashes={p:hashlib.sha256(
                json.dumps(ids,separators=(',',':')).encode()).hexdigest() for p,ids in prompts.items()},
                teacher_sha256=hashlib.sha256(json.dumps(teacher).encode()).hexdigest()))
    path=ROOT/'results/frozen_tokens.json.gz'
    with gzip.open(path,'wt') as stream:json.dump(rows,stream)
    write(target,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=MODELS,requests=manifest,
        sources=previous['sources'],token_artifact_sha256=sha(path),independent_requests_per_model=8,
        domains=dict(coding=3,math=3,document=2),primary_prefixes=[1024,16384],
        confirmation_prefix=32704,confirmation_requests=['coding_00','math_00','document_00'],
        hardware_contract='Same 64 teacher token IDs for every R/TP/family within each model/request. No EOS.',
        prompt_comparability='Fixed token count, not equal token IDs or semantic difficulty across tokenizers.',
        provenance='Same pinned source pools as the previous calibration. Not a new held-out cohort.',
        repeats=3,warmup=3,heldout='NOT_SELECTED. Only after a plausible loop-specific effect.',
        parity=PARITY,quality='Forced continuations and R1/R2 are not benchmark quality. Natural probes remain separate.'))
    print('Frozen 8 requests/model, primary1K/16K, bounded32K confirmation, and64 teacher tokens.',flush=True)


if __name__=='__main__':main()
