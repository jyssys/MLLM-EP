"""Prepare published Part I anchor data with frozen source revisions."""
from collections import defaultdict
import json
from pathlib import Path
import urllib.request
from .common import DATA,ROOT,digest,write,setup

def jsonl(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values)
    if path.exists():
        assert path.read_text()==payload,path
    else:path.write_text(payload)

def download(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix(path.suffix+'.part')
        urllib.request.urlretrieve(url,temporary)
        temporary.replace(path)
    return path

def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    setup()
    prepared=DATA/'paper_part1'
    sources={
        'arc':('allenai/ai2_arc','210d026faf9955653af8916fad021475a3f00453'),
        'mmlu':('cais/mmlu','c30699e8356da336a370243923dbaf21066bb9fe')}
    files=[]
    def parquet(key,name):
        repo,revision=sources[key]
        path=Path(hf_hub_download(repo,name,repo_type='dataset',revision=revision,local_dir=DATA/'downloads'/key))
        files.append(dict(repository=repo,revision=revision,path=name,sha256=digest(path)))
        return pq.read_table(path).to_pylist()
    arc=[]
    for row in parquet('arc','ARC-Challenge/test-00000-of-00001.parquet'):
        choices=row['choices']
        if len(choices['text'])!=4:continue
        answer=list(choices['label']).index(row['answerKey'])
        arc.append(dict(question=row['question'],choices=dict(zip('ABCD',choices['text'])),label='ABCD'[answer]))
    assert len(arc)==1165
    jsonl(prepared/'arc_challenge/test.jsonl',arc)
    unavailable=[]
    try:
        hellaswag=download('https://raw.githubusercontent.com/rowanz/hellaswag/6774d74db0a963013d28bd9323c32de8dd506038/data/hellaswag_val.jsonl',
            DATA/'downloads/hellaswag_val.jsonl')
        assert len(hellaswag.read_text().splitlines())==10042
        destination=prepared/'hellaswag/val.jsonl'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert digest(destination)==digest(hellaswag)
        else:destination.write_bytes(hellaswag.read_bytes())
        files.append(dict(repository='rowanz/hellaswag',revision='6774d74db0a963013d28bd9323c32de8dd506038',
            path='data/hellaswag_val.jsonl',sha256=digest(hellaswag)))
    except Exception as error:
        unavailable.append(dict(task='hellaswag',error=repr(error),status='UNAVAILABLE_OFFICIAL_DISTRIBUTION',
            diagnostic='Official GitHub remote reports DMCA takedown. Official HF repository returns 401. No unofficial mirror substitution.'))
    counts={}
    for split in ['dev','test']:
        by=defaultdict(list)
        for row in parquet('mmlu',f'all/{split}-00000-of-00001.parquet'):
            subject=row['subject']
            by[subject].append(dict(question=row['question'],answer='ABCD'[row['answer']],
                choices=dict(zip('ABCD',row['choices'])),dataset=f'{subject}_{split}'))
        assert len(by)==57
        for subject,items in by.items():
            if split=='dev':assert len(items)==5
            jsonl(prepared/f'mmlu/{subject}/{split}.jsonl',items)
        counts[split]=sum(map(len,by.values()))
    assert counts['test']==14042
    manifest=dict(sources=files,counts=dict(arc_challenge=1165,mmlu=14042,mmlu_dev=counts['dev']),unavailable=unavailable,
        official_task_definitions='xllm/paper_part1/tasks',protocol='paper-part1 task specifications, anchor subset',
        prepared_files=[dict(path=str(p.relative_to(prepared)),sha256=digest(p),rows=p.read_bytes().count(b'\n')) for p in sorted(prepared.rglob('*.jsonl'))])
    write(ROOT/'quality/dataset_manifest.json',manifest)
    print('DATA_PREPARED',manifest['counts'],flush=True)

if __name__=='__main__':main()
