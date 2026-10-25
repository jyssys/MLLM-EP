"""Restore frozen text from public dataset revisions. Check every text hash."""
import hashlib
import json
from pathlib import Path
import urllib.request
import argparse
import pyarrow.parquet as pq
from .common import ROOT,ASSETS,write

def fetch(repo,revision,pattern):
    from huggingface_hub import HfApi,hf_hub_download
    files=sorted(f for f in HfApi().list_repo_files(repo,repo_type='dataset',revision=revision) if pattern(f))
    assert files,(repo,revision)
    result=[]
    for relative in files:
        root=ASSETS/'datasets'/repo.replace('/','--')/revision
        path=Path(hf_hub_download(repo,relative,repo_type='dataset',revision=revision,local_dir=root))
        if path.suffix=='.jsonl':
            result.extend(json.loads(line) for line in path.read_text().splitlines())
        else:
            result.extend(pq.read_table(path).to_pylist())
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--verify-only',action='store_true')
    args=p.parse_args()
    target=ROOT/'workload_manifest.json'
    assert args.verify_only or not target.exists(),'Never overwrite the full local workload manifest'
    public=json.loads((ROOT/'workload_manifest_public.json').read_text())
    sources=public['sources']
    pile=fetch(sources['pile_repository'],sources['pile_revision'],lambda p:p.endswith('.parquet'))
    gsm=fetch(sources['gsm_repository'],sources['gsm_revision'],lambda p:p.startswith('main/') and '/test-' in p and p.endswith('.parquet'))
    math=fetch(sources['math_repository'],sources['math_revision'],lambda p:p=='test.jsonl' or ('test-' in p and p.endswith('.parquet')))
    rows=[]
    stress=['The same word repeats. '*1000,
        ''.join(f'item_{j} = {j} + {j+1}\n' for j in range(1000)),
        'red blue green yellow. '*1000,'Copy these values exactly: 1 7 3 9 2 8. '*1000]
    for saved in public['rows']:
        row=dict(saved)
        index=int(row['id'].split('/')[-1])
        if row['dataset']=='language':
            text=pile[index]['text'][:20000]
        elif row['dataset']=='gsm8k':
            row['question']=gsm[index]['question']
            text='Question: '+row['question']+'\nAnswer:'
        elif row['dataset']=='math500':
            row['question']=math[index]['problem']
            text='Problem: '+row['question']+'\nAnswer:'
        else:
            text=stress[index]
        assert hashlib.sha256(text.encode()).hexdigest()==row['text_sha256'],row['id']
        row['text']=text
        rows.append(row)
    restored=dict(public,rows=rows,reconstruction='Every selected raw-text SHA-256 matches the frozen manifest')
    if not args.verify_only:
        write(target,restored)
    print('RECONSTRUCTED_TEXT_HASHES',len(rows))

if __name__=='__main__':
    main()
