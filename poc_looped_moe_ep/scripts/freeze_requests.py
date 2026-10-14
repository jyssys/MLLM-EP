"""Metadata-only seeded WikiText paragraph episodes, before cohort routing."""
import random
import sys
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from poc_looped_moe_ep.contract import *


def main():
    target=ROOT/'results/frozen_requests.json'
    assert not target.exists(),'Never overwrite the frozen cohort'
    sys.path.append('/home/esjung/external/multibd-poc-deps')
    import pyarrow.parquet as pq
    dataset='Salesforce/wikitext';revision='b08601e04326c79dfdd32d625aee71d232d685c3'
    file='wikitext-2-raw-v1/test-00000-of-00001.parquet'
    local=hf_hub_download(dataset,file,repo_type='dataset',revision=revision)
    texts=pq.read_table(local,columns=['text']).column('text').to_pylist()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,subfolder=SUBFOLDER)
    eligible=[];seen=set()
    for row,text in enumerate(texts):
        if len(text)<1000:continue
        tokens=tokenizer.encode(text,add_special_tokens=False)
        if len(tokens)<176:continue
        digest=__import__('hashlib').sha256(text.encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest);eligible.append((row,text,tokens,digest))
    random.Random(3407).shuffle(eligible)
    requests=[]
    for i,(row,text,tokens,digest) in enumerate(eligible[:32]):
        length=64 if i%2==0 else 128
        requests.append(dict(request_id=f'wt2_{row}',split='calibration' if i<8 else 'held_out',
            source_row=row,text_sha256=digest,prompt_tokens=tokens[:length],
            target_tokens=tokens[length:length+16],prompt_length=length,
            text_excerpt=tokenizer.decode(tokens[:length+16])))
    assert len(requests)==32
    write(target,dict(dataset=dataset,revision=revision,file=file,source_file_sha256=sha(local),
        seed=3407,selection='text length and token count only; no model output/routing inspected',
        prompt_lengths=[64,128],teacher_forced_prefix_offsets=[0,4,8],
        greedy_correctness_tokens=4,native_H=16,requests=requests))
    print('Frozen:',len(requests),'8 calibration / 24 held-out')


if __name__=='__main__':main()
