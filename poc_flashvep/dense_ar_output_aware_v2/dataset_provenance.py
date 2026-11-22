"""Publish request identities and reconstruction metadata without bulk contexts."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
from .common import ROOT,REPO,MODEL,MODEL_REVISION,REVISION,write,digest,preserve

RECONSTRUCTED={'original_context','original_input','reference_answers','rendered_prompt_full','input_ids','all_classes'}
TEMPLATES=REPO/'poc_flashvep/reports/dense_ar_native_pd_real_workload_reproduction_20261119/references/longbench_v1_dataset2prompt.json'

def prepare():
    preserve();target=ROOT/'publication/provenance';target.mkdir(parents=True,exist_ok=True)
    assert not (target/'dataset_rosters_v2.json').exists()
    templates=json.loads(TEMPLATES.read_text());write(target/'official_task_templates_v1.json',templates)
    rosters=[];plain=[]
    for path in sorted((ROOT/'data').glob('*.json')):
        data=json.loads(path.read_text())
        if not isinstance(data,list) or not data or not all(isinstance(r,dict) and 'input_ids' in r and 'prompt_id' in r for r in data):
            plain.append(dict(path=path.name,bytes=path.stat().st_size,SHA256=digest(path)));continue
        rows=[]
        for row in data:
            assert not row.get('context_truncated',False),row['prompt_id']
            assert row['source_revision']==REVISION
            rows.append(dict(keys=list(row),metadata={k:v for k,v in row.items() if k not in RECONSTRUCTED}))
        rosters.append(dict(dataset=path.name,count=len(data),SHA256=digest(path),rows=rows))
    source=REPO/'poc_flashvep/reports/dense_ar_longbench400_2d_20261121/data/candidate_input_inventory.json'
    source_sha=json.loads(source.read_text())['source_zip_SHA256']
    write(target/'dataset_rosters_v2.json',dict(time=time.time(),dataset_revision=REVISION,source_archive_SHA256=source_sha,
        model_revision=MODEL_REVISION,templates_SHA256=digest(TEMPLATES),rosters=rosters,other_data_artifacts=plain,
        scope='Ordered request metadata and hashes preserve every recorded dataset roster. '
            'Bulk third-party contexts and their token sequences are not republished here. '
            'Obtain the pinned official dataset under its source terms. Reconstruct prompts with the pinned templates and tokenizer. '
            'The LongBench code license alone does not establish rights for every underlying context. '
            'Full original data artifacts remain unchanged in the local campaign evidence.'))
    preserve();print('DATASET_PROVENANCE_ROSTERS',len(rosters),flush=True)

def reconstruct(archive,output,selected=None):
    metadata=json.loads((ROOT/'publication/provenance/dataset_rosters_v2.json').read_text())
    archive=Path(archive);output=Path(output)
    assert digest(archive)==metadata['source_archive_SHA256']
    templates=json.loads((ROOT/'publication/provenance/official_task_templates_v1.json').read_text())
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
    rosters=[r for r in metadata['rosters'] if selected is None or r['dataset']==selected]
    assert rosters
    wanted={r['metadata']['prompt_id'] for roster in rosters for r in roster['rows']};originals={}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if not member.endswith('.jsonl') or Path(member).stem.endswith('_e'):continue
            task=Path(member).stem
            for line in bundle.read(member).decode().splitlines():
                row=json.loads(line);pid=f'longbench_v1_{task}_{row["_id"]}'
                if pid in wanted:originals[pid]=(task,row)
    assert originals.keys()==wanted
    cache={};results=[]
    for roster in rosters:
        rows=[]
        for item in roster['rows']:
            pid=item['metadata']['prompt_id'];task,original=originals[pid]
            if pid not in cache:
                prompt=templates[task].format(**original)
                ids=tokenizer.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False,return_dict=False)
                assert hashlib.sha256(prompt.encode()).hexdigest()==item['metadata']['prompt_text_SHA256']
                cache[pid]=dict(original_context=original['context'],original_input=original['input'],
                    reference_answers=original['answers'],rendered_prompt_full=prompt,input_ids=dict(qwen8=ids),all_classes=original.get('all_classes'))
            values=dict(item['metadata'],**cache[pid]);row={key:values[key] for key in item['keys']}
            assert len(row['input_ids']['qwen8'])==row['input_tokens']
            rows.append(row)
        path=output/roster['dataset'];assert not path.exists(),path
        write(path,rows);actual=digest(path);assert actual==roster['SHA256'],(roster['dataset'],actual,roster['SHA256'])
        results.append(dict(dataset=roster['dataset'],count=len(rows),SHA256=actual,byte_exact=True))
        print('BYTE_EXACT_DATASET_RECONSTRUCTION',roster['dataset'],len(rows),flush=True)
    write(output/'reconstruction_validation_v2.json',dict(time=time.time(),results=results,
        source_archive_SHA256=digest(archive),model_revision=MODEL_REVISION,passed=True))
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','reconstruct'))
    p.add_argument('--archive');p.add_argument('--output');p.add_argument('--dataset');args=p.parse_args()
    if args.action=='prepare':prepare()
    else:reconstruct(args.archive,args.output,args.dataset)
