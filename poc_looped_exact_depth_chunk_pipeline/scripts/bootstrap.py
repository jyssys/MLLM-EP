"""Pin objective questions, prompt tokens, settings, and current environment."""
import datetime
import importlib.metadata
import shutil
import subprocess
import sys
import urllib.request
from collections import defaultdict
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    assert not (ROOT/'results/environment.json').exists(),'Preserve an earlier freeze'
    ASSETS.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SPEC,ROOT/'authoritative_spec.md')
    original=Path('/home/esjung/MLLM-EP-github')
    environment=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        spec_sha256=sha(SPEC),GPUS=GPUS,UUIDS=UUIDS,model_revision=REVISION,
        model_files={p.name:sha(p) for p in MODEL.iterdir() if p.suffix in ['.py','.json']},
        nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        original_diffs={k:hashlib.sha256(subprocess.check_output(cmd,cwd=original)).hexdigest() for k,cmd in
            [('staged',['git','diff','--cached']),('unstaged',['git','diff'])]})
    environment['runtime']=json.loads(subprocess.check_output([PYTHON,'-c',
        'import importlib.metadata as m,json;print(json.dumps({p:m.version(p) for p in ["vllm","torch","transformers"]}))'],text=True))
    write(ROOT/'results/environment.json',environment)
    sys.path.insert(0,str(ASSETS/'cpu_deps'))
    import pyarrow.parquet as pq
    source=[];rows=[]
    for name,info in DATASETS.items():
        url=f'https://huggingface.co/datasets/{info["repo"]}/resolve/{info["revision"]}/{info["file"]}'
        target=ASSETS/(name+Path(info['file']).suffix)
        if not target.exists():urllib.request.urlretrieve(url,target)
        values=pq.read_table(target).to_pylist() if target.suffix=='.parquet' else [json.loads(s) for s in target.read_text().splitlines()]
        source.append(dict(name=name,**info,url=url,path=str(target),sha256=sha(target),available_rows=len(values)))
        if name=='math100':
            grouped=defaultdict(list)
            for value in values:grouped[value['subject']].append(value)
            for group in grouped.values():group.sort(key=lambda r:hashlib.sha256(str(r['unique_id']).encode()).hexdigest())
            selected=[]
            while len(selected)<100:
                for subject in sorted(grouped):
                    if grouped[subject] and len(selected)<100:selected.append(grouped[subject].pop(0))
            values=selected
        assert len(values)==(30 if name=='aime24' else 100)
        for index,value in enumerate(values):
            problem=value['problem'];answer=str(value['answer'])
            identifier=str(value.get('id',value.get('unique_id',index)))
            prompt='Solve the following problem. Show your reasoning and give the final answer in \\boxed{...}.\n\n'+problem
            rows.append(dict(dataset=name,id=identifier,index=index,problem=problem,answer=answer,prompt=prompt,
                subject=value.get('subject'),level=value.get('level'),source_revision=info['revision']))
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
    for row in rows:
        row['prompt_ids']=tokenizer.apply_chat_template([dict(role='user',content=row['prompt'])],tokenize=True,add_generation_prompt=True)
        row['prompt_sha256']=hashlib.sha256(json.dumps(row['prompt_ids']).encode()).hexdigest()
    write(ROOT/'results/frozen_quality_questions.json',rows)
    write(ROOT/'results/quality_contract.json',dict(model_revision=REVISION,R=4,early_exit_threshold=1.,seeds=SEEDS,
        sampling=dict(temperature=1.,top_p=.7,top_k=-1,repetition_penalty=1.),greedy=dict(temperature=0.,top_p=1.),
        primary_cap=4096,upgrade_cap=8192,upgrade_rule='If >=10% of AIME sampling or greedy outputs reach cap, rerun the complete AIME lane at 8192. Preserve both caps.',
        secondary_rule='100 metadata-stratified MATH-500 problems, greedy. Upgrade the complete secondary lane if >=10% reach cap.',
        quality_completion_gate='At final cap, >=90% must terminate naturally in each primary lane. Otherwise QUALITY_BASELINE_WEAK.',
        protocol_comparison='REFERENCE_PROTOCOL_NOT_IDENTICAL',request_batch=8,chat_template=tokenizer.chat_template,
        special_tokens=tokenizer.special_tokens_map,eos_token_id=tokenizer.eos_token_id,tokenizer_class=type(tokenizer).__name__,
        datasets=source,component_tolerance=TOLERANCE,selection='Hash order within subject, then subject round-robin. No output-based selection.'))
    # Preserve exact official source text outside Git. HTTP failures are evidence.
    fetched=[]
    for name,url in [('paper','https://arxiv.org/html/2510.25741v2'),('project','https://ouro-llm.github.io/'),
                     ('official_repo','https://api.github.com/repos/ByteDance/Ouro')]:
        try:
            data=urllib.request.urlopen(url,timeout=30).read();path=ASSETS/(name+'.txt');path.write_bytes(data)
            fetched.append(dict(name=name,url=url,path=str(path),sha256=sha(path),status='FETCHED'))
        except Exception as error:fetched.append(dict(name=name,url=url,status='UNAVAILABLE',error=repr(error)))
    write(ROOT/'results/official_sources.json',fetched)
    print('Frozen',len(rows),'quality problems. No GPU execution.')


if __name__=='__main__':main()
