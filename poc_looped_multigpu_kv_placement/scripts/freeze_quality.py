"""Freeze one calibration case per domain with each official chat template."""
import datetime
import gzip
from poc_looped_multigpu_kv_placement.contract import *


def main():
    path=ROOT/'results/frozen_quality.json';assert not path.exists()
    from transformers import AutoTokenizer
    with gzip.open(ROOT/'results/frozen_tokens.json.gz','rt') as stream:source=json.load(stream)
    cases=[]
    for name in MODELS:
        tokenizer=AutoTokenizer.from_pretrained(model_path(name),trust_remote_code=True)
        for rid in ['coding_00','math_00','document_00']:
            row=next(r for r in source if r['model']==name and r['request_id']==rid)
            text=tokenizer.decode(row['prefix_token_ids']['1024'],skip_special_tokens=False)
            ids=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=True,add_generation_prompt=True)
            assert len(ids)+512<(32768 if name=='ouro' else 4096)
            cases.append(dict(model=name,request_id=rid,prompt_token_ids=ids,prompt_sha256=hashlib.sha256(text.encode()).hexdigest()))
    write(path,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),max_new_tokens=512,natural_EOS=True,
        selection='First calibration ID per domain. Selected by metadata only. No output inspection.',cases=cases,
        scope='Three natural-EOS sanity cases per model. Not official benchmark accuracy or a held-out quality evaluation.'))
    print('Frozen three natural-EOS calibration sanity cases per model.')


if __name__=='__main__':main()
