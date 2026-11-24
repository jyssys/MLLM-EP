"""Profile original image requests before any serving-performance selection."""
import ast
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import types

sys.path.insert(0,'/home/esjung/external/mllm-epd-prefill-heavy-deps-20261124')
from .common import ROOT,MODEL_ID,write,digest,initialize,preserve

SEED=2026112401
LIMIT=32768
OUTPUT_CAP=2048
TEMPLATE_VERSION=2
CALIBRATION=dict(single_image=4,multi_image=8,multi_page=8,long_document=12)
TARGET=dict(single_image=50,multi_image=100,multi_page=100,long_document=150)


def snapshot(repo):
    return json.loads((ROOT/'state'/('download_'+repo.replace('/','__')+'.json')).read_text())


def source_rows(repo,pattern,media_columns,id_key,stratum):
    import pyarrow.parquet as pq
    record=snapshot(repo)
    for file in sorted(Path(record['directory']).glob(pattern)):
        pf=pq.ParquetFile(file)
        columns=[x for x in pf.schema_arrow.names if x not in media_columns]
        for number,row in enumerate(pf.read(columns=columns).to_pylist()):
            original=str(row[id_key])
            yield dict(request_id=repo.replace('/','__')+'__'+original,source=repo,
                source_revision=record['revision'],original_id=original,metadata=row,
                parquet=str(file),row_index=number,media_columns=media_columns,stratum=stratum)


def pools():
    single=list(source_rows('lmms-lab/DocVQA','DocVQA/*.parquet',['image'],'questionId','single_image'))
    multi=list(source_rows('TIGER-Lab/Mantis-Eval','mantis_eval/*.parquet',['images'],'id','multi_image'))
    multi+=list(source_rows('MUIRBENCH/MUIRBENCH','data/*.parquet',['image_list'],'idx','multi_image'))
    pages=list(source_rows('lmms-lab/MP-DocVQA','data/val-*.parquet',
        ['image_'+str(i) for i in range(1,21)],'questionId','multi_page'))
    long=list(source_rows('yubo2333/MMLongBench-Doc','data/*.parquet',[],
        'question','long_document'))
    for number,row in enumerate(long):
        row['original_id']=str(number)
        row['request_id']='yubo2333__MMLongBench-Doc__'+str(number)
    results=dict(single_image=single,multi_image=multi,multi_page=pages,long_document=long)
    for name,rows in results.items():
        # Freeze source priority first. Use a deterministic permutation within each source.
        for r in rows:
            r['order_key']=hashlib.sha256((str(SEED)+r['request_id']).encode()).hexdigest()
        priority={'TIGER-Lab/Mantis-Eval':0,'MUIRBENCH/MUIRBENCH':1}
        rows.sort(key=lambda r:(priority.get(r['source'],0),r['order_key']))
    return results


@lru_cache(maxsize=3)
def media_table(file,columns):
    import pyarrow.parquet as pq
    return pq.read_table(file,columns=list(columns))


def put_image(blob,original_name=None):
    from PIL import Image
    image=Image.open(io.BytesIO(blob));image.load()
    raw_hash=hashlib.sha256(blob).hexdigest()
    pixel_hash=hashlib.sha256(str(image.size).encode()+image.convert('RGB').tobytes()).hexdigest()
    extension={'JPEG':'.jpg','PNG':'.png','TIFF':'.tiff','WEBP':'.webp'}.get(image.format,'.img')
    path=ROOT/'media'/raw_hash[:2]/(raw_hash+extension)
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        temporary.write_bytes(blob);os.replace(temporary,path)
    return dict(path=str(path),SHA256=raw_hash,pixel_SHA256=pixel_hash,
        original_resolution=list(image.size),original_name=original_name)


def render_document(doc_id):
    import fitz
    receipt=snapshot('yubo2333/MMLongBench-Doc')
    document=Path(receipt['directory'])/'documents'/doc_id
    assert document.is_file(),document
    sha=digest(document);cache=ROOT/'data/rendered_documents'/(sha+'.json')
    if cache.exists():return json.loads(cache.read_text())
    pdf=fitz.open(document);images=[]
    for i,page in enumerate(pdf):
        pix=page.get_pixmap(dpi=72,colorspace=fitz.csRGB,alpha=False)
        images.append(put_image(pix.tobytes('png'),f'{doc_id}:page_{i+1}'))
    result=dict(document=str(document),document_SHA256=sha,page_count=len(images),
        rendering=dict(dpi=72,colorspace='RGB',all_pages=True,crop=False),images=images)
    write(cache,result)
    return result


def load_media(row):
    meta=row['metadata'];source=row['source'];doc=None
    if source=='yubo2333/MMLongBench-Doc':
        doc=render_document(meta['doc_id']);images=doc['images']
        document_id=meta['doc_id'];references=[meta['answer']]
    else:
        table=media_table(row['parquet'],tuple(row['media_columns']))
        record=table.slice(row['row_index'],1).to_pylist()[0]
        if source=='lmms-lab/MP-DocVQA':
            media=[record.get('image_'+str(i)) for i in range(1,21)]
            references=ast.literal_eval(meta['answers']);document_id=meta['doc_id']
        elif source=='TIGER-Lab/Mantis-Eval':
            media=record['images'];references=[meta['answer']];document_id=None
        elif source=='MUIRBENCH/MUIRBENCH':
            media=record['image_list'];references=[meta['answer']];document_id=None
        else:
            media=[record['image']];references=meta['answers'];document_id=meta.get('ucsf_document_id')
        images=[put_image(m['bytes'],m.get('path')) for m in media if m is not None and m.get('bytes')]
    assert images,'A request must contain at least one original image'
    if row['stratum'] in ('multi_image','multi_page'):assert len(images)>=2
    question=meta['question']
    options=meta.get('options') or []
    # Preserve options. Do not add an instruction that changes the expected answer length.
    if options:
        rendered=[str(v) if re.match(r'^\(?[A-Z][).]',str(v)) else f'({chr(65+i)}) {v}'
            for i,v in enumerate(options)]
        question+='\n'+'\n'.join(rendered)
    return dict(images=images,document=doc,document_id=document_id,
        question_original=meta['question'],question=question,reference_answers=references,
        options=options,page_count=len(images) if row['stratum'] in ('multi_page','long_document') else None)


@lru_cache(maxsize=1)
def processor():
    import torch
    torch.set_num_threads(1)
    from transformers import AutoProcessor,AutoConfig
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
    receipt=json.loads((ROOT/'state/model_download_complete.json').read_text())
    model=receipt['directory']
    proc=AutoProcessor.from_pretrained(model)
    instance=types.SimpleNamespace(config=AutoConfig.from_pretrained(model))
    instance.get_vision_position_ids=types.MethodType(Qwen2_5_VLModel.get_vision_position_ids,instance)
    return proc,instance,Qwen2_5_VLModel.get_rope_index


def profile(row):
    import numpy as np
    import torch
    from PIL import Image
    output=ROOT/'data/request_profiles'/(hashlib.sha256((row['request_id']+f':template{TEMPLATE_VERSION}').encode()).hexdigest()+'.json')
    if output.exists():return json.loads(output.read_text())
    started=time.time()
    try:
        material=load_media(row)
        proc,rope,rope_method=processor()
        images=[Image.open(v['path']).convert('RGB') for v in material['images']]
        messages=[dict(role='user',content=[dict(type='image') for _ in images]+[dict(type='text',text=material['question'])])]
        prompt=proc.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        encoded=proc(text=[prompt],images=images,return_tensors='pt',return_mm_token_type_ids=True)
        ids=encoded['input_ids'][0];length=len(ids)
        grid=encoded['image_grid_thw'].tolist();visual=int((ids==proc.image_token_id).sum())
        special_set=set(proc.tokenizer.all_special_ids)-{proc.image_token_id}
        special=sum(int(t) in special_set for t in ids)
        spans=[];start=None
        for i,t in enumerate(ids.tolist()+[-1]):
            if t==proc.image_token_id and start is None:start=i
            if t!=proc.image_token_id and start is not None:
                spans.append([start,i]);start=None
        assert len(spans)==len(images)==len(grid),(len(spans),len(images),len(grid))
        assert sum(b-a for a,b in spans)==visual
        for (a,b),g,media in zip(spans,grid,material['images']):
            assert b-a==int(np.prod(g))//4
            media['grid_thw']=g;media['processed_resolution']=[g[2]*14,g[1]*14]
            media['visual_span']=[a,b];media['visual_tokens']=b-a
        if 'mm_token_type_ids' not in encoded:
            encoded['mm_token_type_ids']=(encoded['input_ids']==proc.image_token_id).long()
        positions,delta=rope_method(rope,input_ids=encoded['input_ids'],
            mm_token_type_ids=encoded['mm_token_type_ids'],image_grid_thw=encoded['image_grid_thw'],
            attention_mask=encoded.get('attention_mask'))
        tensor_path=ROOT/'data/request_tensors'/(output.stem+'.npz')
        tensor_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(tensor_path,input_ids=ids.numpy(),image_grid_thw=np.array(grid),
            positions=positions.numpy(),rope_delta=delta.numpy(),
            mm_token_type_ids=encoded['mm_token_type_ids'].numpy())
        result=dict(**row,**material,prompt=prompt,prompt_template_version=TEMPLATE_VERSION,
            prompt_SHA256=hashlib.sha256(prompt.encode()).hexdigest(),
            model=MODEL_ID,model_revision=json.loads((ROOT/'state/model_download_complete.json').read_text())['revision'],
            processor_revision=json.loads((ROOT/'state/model_download_complete.json').read_text())['revision'],
            total_input_tokens=length,visual_tokens=visual,special_tokens=special,text_tokens=length-visual-special,
            image_count=len(images),image_visual_spans=spans,grid_thw=grid,
            token_metadata_file=str(tensor_path),token_metadata_SHA256=digest(tensor_path),
            encoder_output_shape_structural=[visual,3584],encoder_output_shape_measured=None,
            natural_output_tokens=None,natural_output_status='NOT_MEASURED',
            context_limit=LIMIT,output_reserve=OUTPUT_CAP,truncated=False,
            eligible=length+OUTPUT_CAP<=LIMIT,exclusion_reason=None if length+OUTPUT_CAP<=LIMIT else 'CONTEXT_OVERFLOW',
            elapsed_seconds=time.time()-started,evidence='PROCESSOR_MEASUREMENT')
    except Exception as error:
        result=dict(**row,eligible=False,exclusion_reason=type(error).__name__+': '+str(error),
            evidence='FAILED_EXECUTION',elapsed_seconds=time.time()-started)
    write(output,result)
    return result


def keys(row):
    result=set()
    for image in row.get('images',[]):
        result.add('sha:'+image['SHA256']);result.add('pixel:'+image['pixel_SHA256'])
    if row.get('document_id'):
        # Include the source-independent document identifier for DocVQA / MP-DocVQA overlap.
        result.add('document:'+str(row['document_id']))
    if row.get('document'):
        result.add('pdf:'+row['document']['document_SHA256'])
    return result


def group_split(rows):
    parents=list(range(len(rows)));owners={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,r in enumerate(rows):
        for key in keys(r):
            if key in owners:parents[find(i)]=find(owners[key])
            else:owners[key]=i
    groups={}
    for i,r in enumerate(rows):groups.setdefault(find(i),[]).append(r)
    groups=sorted(groups.values(),key=lambda group:hashlib.sha256((str(SEED)+min(r['request_id'] for r in group)).encode()).hexdigest())
    reachable={0:[]}
    for i,g in enumerate(groups):
        for total,chosen in sorted(list(reachable.items()),reverse=True):
            size=total+len(g)
            if size<=len(rows)//2 and size not in reachable:reachable[size]=chosen+[i]
    best=max(reachable);chosen=set(reachable[best])
    return [r for i,g in enumerate(groups) if i in chosen for r in g],\
        [r for i,g in enumerate(groups) if i not in chosen for r in g],list(map(len,groups))


def main():
    initialize()
    (ROOT/'data/request_profiles').mkdir(parents=True,exist_ok=True)
    results=pools()
    write(ROOT/'data/candidate_pool.json',dict(seed=SEED,sources=results,
        counts={name:len(rows) for name,rows in results.items()},scope='All metadata candidates. No output-length or performance filtering.'))
    selected=[];calibration=[];calibration_keys=set();main_keys=set();seen=set();audit=[]
    with ProcessPoolExecutor(max_workers=4) as executor:
        for stratum,rows in results.items():
            needed=TARGET[stratum];cal_needed=CALIBRATION[stratum]
            pointer=0;accepted=[];cal=[]
            while pointer<len(rows) and (len(accepted)<needed or len(cal)<cal_needed):
                batch=rows[pointer:pointer+8];pointer+=len(batch)
                for row in executor.map(profile,batch):
                    audit.append(dict(request_id=row['request_id'],eligible=row['eligible'],
                        reason=row.get('exclusion_reason'),total_input_tokens=row.get('total_input_tokens'),
                        stratum=stratum,elapsed_seconds=row['elapsed_seconds']))
                    if not row['eligible'] or row['request_id'] in seen:continue
                    group=keys(row)
                    if len(cal)<cal_needed and not group.intersection(calibration_keys|main_keys):
                        cal.append(row);calibration_keys.update(group);seen.add(row['request_id'])
                    elif len(accepted)<needed and not group.intersection(calibration_keys):
                        accepted.append(row);seen.add(row['request_id']);main_keys.update(group)
                write(ROOT/'state/dataset_build_progress.json',dict(time=time.time(),stratum=stratum,
                    inspected=pointer,pool=len(rows),main=len(accepted),main_target=needed,
                    calibration=len(cal),calibration_target=cal_needed,complete=False))
                print('PROFILE_PROGRESS',stratum,pointer,len(accepted),len(cal),flush=True)
            selected+=accepted;calibration+=cal
            write(ROOT/'data'/('selected_'+stratum+'.json'),dict(main=accepted,calibration=cal,
                inspected=pointer,pool_count=len(rows),target=needed,shortfall=needed-len(accepted)))
    # Do not fabricate a full target if a stratum exhausts its original candidates.
    # A fallback may run only after a separate documented source-availability decision.
    discovery,heldout,groups=group_split(selected)
    assert not set.union(set(),*(keys(r) for r in discovery)).intersection(set.union(set(),*(keys(r) for r in heldout)))
    assert not set.union(set(),*(keys(r) for r in selected)).intersection(calibration_keys)
    main_manifest=dict(name='MLLM-PrefillHeavy400-v1',seed=SEED,requests=selected,
        calibration_ids=[r['request_id'] for r in calibration],discovery_ids=[r['request_id'] for r in discovery],
        heldout_ids=[r['request_id'] for r in heldout],context_components=groups,
        target_count=400,actual_count=len(selected),selection_basis='source strata and context feasibility only',
        natural_output_status='NOT_MEASURED',performance_selection=False,truncation=False)
    write(ROOT/'data/main_manifest.json',main_manifest)
    write(ROOT/'data/calibration_manifest.json',dict(requests=calibration,seed=SEED))
    write(ROOT/'data/feasibility_audit.json',dict(requests=audit))
    write(ROOT/'state/dataset_input_freeze.json',dict(time=time.time(),main_SHA256=digest(ROOT/'data/main_manifest.json'),
        calibration_SHA256=digest(ROOT/'data/calibration_manifest.json'),count=len(selected),
        discovery=len(discovery),heldout=len(heldout),calibration=len(calibration),
        status='INPUT_FROZEN' if len(selected)==400 else 'SOURCE_FEASIBILITY_SHORTFALL',
        natural_output_status='NOT_MEASURED'))
    preserve()


if __name__=='__main__':main()
