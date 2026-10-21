"""Freeze native-frontend token IDs before further cross-version quality runs."""
import transformers
from transformers import AutoTokenizer
from .contract import *


def main():
    path=ROOT/'frozen_prompt_ids.json';assert not path.exists()
    tok=AutoTokenizer.from_pretrained(str(MODEL),trust_remote_code=True,local_files_only=True)
    rows=read(ROOT/'dataset_manifest.json')['rows'];mapping={r['id']:tok.encode(r['prompt'],add_special_tokens=False) for r in rows}
    verified=0
    for label in ['native_TP1_pilot_v1','native_TP1_pilot_v2','native_TP4_pilot_v2']:
        for file in (ROOT/'raw'/label).glob('*.json'):
            r=read(file)
            if r.get('id') in mapping:
                assert r['prompt_ids']==mapping[r['id']],file
                verified+=1
    write(path,dict(tokenizer_revision=REVISION,transformers=transformers.__version__,BOS=False,rows=mapping,
        verified_native_task_records=verified,scope='Same frozen strings and checkpoint. Explicit IDs prevent tokenizer-library whitespace differences.'))
    print('FROZEN_TOKEN_IDS',len(mapping),verified,flush=True)


if __name__=='__main__':main()
