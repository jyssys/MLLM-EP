"""Verify frozen source media, processor metadata and disjoint request contexts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .common import ROOT,digest,preserve,write
from .dataset_build import keys
from .token_discovery_analysis import table


def assert_disjoint(groups):
    seen_ids=set();seen_context=set();summary=[]
    for name,rows in groups:
        ids={r['request_id'] for r in rows}
        assert len(ids)==len(rows) and not ids&seen_ids
        context=set().union(*(keys(row) for row in rows)) if rows else set()
        assert not context&seen_context, name+' shares an image, pixel content or document'
        summary.append(dict(name=name,requests=len(rows),unique_context_keys=len(context)))
        seen_ids.update(ids);seen_context.update(context)
    return summary


def validate_metadata(row,values,image_token_id,reference_ids):
    length=row['total_input_tokens'];ids=values['input_ids'];grid=values['image_grid_thw'];positions=values['positions']
    assert ids.shape==(length,) and np.array_equal(ids,np.asarray(reference_ids))
    assert positions.shape==(3,1,length) and np.issubdtype(positions.dtype,np.integer)
    assert grid.shape==(row['image_count'],3) and np.array_equal(grid,np.asarray(row['grid_thw']))
    assert values['rope_delta'].shape==(1,1)
    assert np.array_equal(values['mm_token_type_ids'],(ids==image_token_id)[None,:])
    assert row['total_input_tokens']==row['visual_tokens']+row['text_tokens']+row['special_tokens']
    spans=row['image_visual_spans'];assert len(spans)==row['image_count']==len(row['images'])
    assert sum(b-a for a,b in spans)==np.count_nonzero(ids==image_token_id)==row['visual_tokens']
    previous=0
    for index,((begin,end),media) in enumerate(zip(spans,row['images'])):
        assert previous<=begin<end<=length and np.all(ids[begin:end]==image_token_id)
        assert end-begin==int(np.prod(grid[index]))//4==media['visual_tokens']
        assert media['visual_span']==[begin,end]
        assert media['processed_resolution']==[int(grid[index,2])*14,int(grid[index,1])*14]
        previous=end
    assert not row['truncated'] and row['eligible'] and length+row['output_reserve']<=row['context_limit']
    assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_SHA256']
    assert row['question_original']==row['metadata']['question']
    return dict(request_id=row['request_id'],source=row['source'],source_revision=row['source_revision'],
        input_tokens=length,visual_tokens=row['visual_tokens'],text_tokens=row['text_tokens'],special_tokens=row['special_tokens'],
        image_count=row['image_count'],page_count=row['page_count'],metadata_SHA256=row['token_metadata_SHA256'],
        prompt_SHA256=row['prompt_SHA256'],reference_prompt_ids='STRICT_MATCH',MRoPE_file='HASH_AND_SHAPE_PASS',
        encoder_shape_source='STRUCTURAL_ACCOUNTING in frozen manifest; separate captured GPU correctness evidence is not generalized here')


def check_file(path,expected_sha,expected_bytes=None):
    path=Path(path);assert path.is_file(),str(path)
    if expected_bytes is not None:assert path.stat().st_size==expected_bytes,str(path)
    actual=digest(path);assert actual==expected_sha,str(path)
    return dict(path=str(path),bytes=path.stat().st_size,SHA256=actual,passed=True)


def main():
    preserve();report=ROOT/'analysis/dataset_integrity_audit_v0.json';assert not report.exists()
    begin=time.time();main_path=ROOT/'data/main_manifest_template_v2.json'
    subset_path=ROOT/'data/main100_manifest_template_v2.json';cal_path=ROOT/'data/calibration_manifest_template_v2.json'
    main=json.loads(main_path.read_text());subset=json.loads(subset_path.read_text());cal=json.loads(cal_path.read_text())
    freeze=json.loads((ROOT/'state/dataset_final_input_freeze.json').read_text())
    reduced=json.loads((ROOT/'state/dataset_main100_freeze.json').read_text())
    assert digest(main_path)==freeze['main_SHA256']==subset['parent_SHA256']
    assert digest(cal_path)==freeze['calibration_SHA256']
    assert digest(subset_path)==reduced['main_SHA256'] and digest(ROOT/'configs/main100_selection_plan.json')==subset['selection_plan_SHA256']
    assert len(main['requests'])==400 and len(subset['requests'])==100 and len(cal['requests'])==32
    parent_by_id={r['request_id']:r for r in main['requests']}
    assert all(row==parent_by_id[row['request_id']] for row in subset['requests']), 'Subset request content changed'
    rows=main['requests']+cal['requests'];by_id={r['request_id']:r for r in rows};assert len(by_id)==432
    parent_groups=[('Discovery200',[r for r in main['requests'] if r['request_id'] in main['discovery_ids']]),
        ('Heldout200',[r for r in main['requests'] if r['request_id'] in main['heldout_ids']]),('Calibration32',cal['requests'])]
    split=assert_disjoint(parent_groups);assert [r['requests'] for r in split]==[200,200,32]
    subset_groups=[('Discovery50',[r for r in subset['requests'] if r['request_id'] in subset['discovery_ids']]),
        ('Heldout50',[r for r in subset['requests'] if r['request_id'] in subset['heldout_ids']]),('Calibration32',cal['requests'])]
    reduced_split=assert_disjoint(subset_groups);assert [r['requests'] for r in reduced_split]==[50,50,32]
    references={}
    for name in ['reference_main400_natural_template_v2_v0','reference_calibration32_template_v2_v0']:
        for line in (ROOT/'raw'/f'{name}_requests.jsonl').open():
            value=json.loads(line);assert value['request_id'] not in references and value['status']=='MEASURED_ONLINE'
            references[value['request_id']]=value
    assert set(references)==set(by_id)
    model_receipt=ROOT/'state/model_download_complete.json';model=json.loads(model_receipt.read_text())
    sources={source:json.loads((ROOT/'state'/('download_'+source.replace('/','__')+'.json')).read_text())
        for source in {r['source'] for r in rows}}
    requests=[];media={};files={};documents={}
    for row in rows:
        assert row['model_revision']==row['processor_revision']==model['revision']
        assert row['source_revision']==sources[row['source']]['revision']
        assert references[row['request_id']]['prompt_SHA256']==row['prompt_SHA256']
        metadata=Path(row['token_metadata_file']);check_file(metadata,row['token_metadata_SHA256'])
        with np.load(metadata,allow_pickle=False) as values:
            requests.append(validate_metadata(row,values,model['config']['image_token_id'],references[row['request_id']]['prompt_token_ids']))
        for image in row['images']:
            previous=media.setdefault(image['path'],image)
            assert (previous['SHA256'],previous['pixel_SHA256'])==(image['SHA256'],image['pixel_SHA256'])
        parquet=Path(row['parquet']);receipt=sources[row['source']]
        names={str(Path(receipt['directory'])/entry['name']):entry for entry in receipt['files']}
        expected=names[str(parquet)];files[str(parquet)]=dict(path=str(parquet),SHA256=expected['SHA256'],bytes=expected['bytes'])
        if row.get('document'):
            document=row['document'];documents[document['document']]=document['document_SHA256']
    # Pixels verify original file contents. No image resizing, page removal or
    # processor rerun is used to alter the frozen request contract.
    from PIL import Image
    images=[]
    for path,record in sorted(media.items()):
        checked=check_file(path,record['SHA256'])
        with Image.open(path) as image:
            image.load();assert list(image.size)==record['original_resolution']
            pixel=hashlib.sha256(str(image.size).encode()+image.convert('RGB').tobytes()).hexdigest()
        assert pixel==record['pixel_SHA256'],path
        images.append(dict(**checked,pixel_SHA256=pixel))
    source_files=[check_file(row['path'],row['SHA256'],row['bytes']) for row in files.values()]
    pdfs=[check_file(path,sha) for path,sha in documents.items()]
    model_files=[check_file(row['path'],row['SHA256'],row['bytes']) for row in model['files']]
    table(ROOT/'analysis/dataset_integrity_requests_v0.csv',requests)
    table(ROOT/'analysis/dataset_integrity_media_v0.csv',images)
    table(ROOT/'analysis/dataset_integrity_source_files_v0.csv',source_files+pdfs)
    table(ROOT/'analysis/dataset_integrity_model_files_v0.csv',model_files)
    write(report,dict(time=time.time(),begin=begin,end=time.time(),status='FROZEN_PROVENANCE_AND_METADATA_REVERIFIED',
        passed=True,main_manifest_SHA256=digest(main_path),main100_manifest_SHA256=digest(subset_path),
        calibration_SHA256=digest(cal_path),model_revision=model['revision'],requests=432,subset_unchanged=True,
        parent_split=split,main100_split=reduced_split,unique_media=len(images),media_bytes=sum(r['bytes'] for r in images),
        model_files=len(model_files),model_bytes=sum(r['bytes'] for r in model_files),source_parquets=len(source_files),
        original_PDFs=len(pdfs),source_counts=dict(Counter(r['source'] for r in main['requests'])),
        actual_encoder_shapes_in_frozen_manifest=sum(r['encoder_output_shape_measured'] is not None for r in rows),
        encoder_shape_scope='Original frozen structural shapes remain structural. This audit does not create a GPU measurement.',
        scope='CPU byte, pixel, prompt-token, metadata and context-ownership audit. Not new model execution, quality PASS, GPU timing or serving gain.'))
    preserve();print('DATASET_PROVENANCE_REVERIFIED',len(requests),len(images),len(model_files),flush=True)


if __name__=='__main__':main()
