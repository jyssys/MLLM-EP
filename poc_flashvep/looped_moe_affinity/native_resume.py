"""Recover complete native cells without altering their original evidence."""
import gzip
import json
from pathlib import Path
import shutil
from .common import ROOT,digest,write

def cell_files(batch,length):
    result=[f'native_forward_b{batch}_p{length}.rank0.json',f'native_generation_b{batch}_p{length}.rank0.jsonl']
    for split in ['calibration','heldout']:
        result.extend([f'full_routes_b{batch}_p{length}_{split}.json',f'full_routes_b{batch}_p{length}_{split}.npz'])
    return result

def complete_cell(directory,batch,length):
    if not all((directory/name).is_file() for name in cell_files(batch,length)):return False
    try:
        forward=json.loads((directory/cell_files(batch,length)[0]).read_text())
        values=[json.loads(line) for line in (directory/cell_files(batch,length)[1]).read_text().splitlines()]
        return len(forward['clean_ms'])==5 and len(values)==3 and all(v['warmup_exact'] and len(v['generated_ids'])==32
            and all(len(ids)==batch for ids in v['generated_ids']) for v in values)
    except (OSError,KeyError,ValueError):return False

def inherit(source,target,route_output):
    source=Path(source)
    assert source.parent==(ROOT/'raw') and source!=target
    completed={(b,length) for b in [1,2,4,8,16] for length in [128,512,2048] if complete_cell(source,b,length)}
    files=[]
    for b,length in sorted(completed):
        for name in cell_files(b,length):
            old=source/name;new=target/name;assert not new.exists()
            shutil.copy2(old,new)
            assert digest(new)==digest(old),'Inherited evidence must remain byte-identical.'
            files.append(dict(path=name,source=old.relative_to(ROOT).as_posix(),sha256=digest(old)))
    with gzip.open(source/'decode_router_corpus.jsonl.gz','rt') as stream:
        for line in stream:
            row=json.loads(line)
            if (row['batch'],row['original_prefix_tokens']) in completed:route_output.write(line)
    route_output.flush()
    write(target/'inherited_evidence.json',dict(source=source.relative_to(ROOT).as_posix(),cells=sorted(completed),files=files,
        source_allocator='DEFAULT',new_allocator='Recorded in this attempt config. Only missing cells use the new process.',
        evidence='BYTE_IDENTICAL_RECOVERY_OF_COMPLETE_CELLS_NOT_REMEASURED'))
    return completed

def directory(model):
    attempts=sorted((ROOT/'raw').glob(f'{model}_native_full_profile_v*'),reverse=True)
    complete=[p for p in attempts if (p/'complete.json').exists()]
    return complete[0] if complete else ROOT/'raw'/f'{model}_native_full_profile_v1'

def provenance(base,batch,length):
    inherited=base/'inherited_evidence.json'
    if inherited.exists():
        data=json.loads(inherited.read_text())
        if [batch,length] in data['cells']:
            return provenance(ROOT/data['source'],batch,length)
    config=ROOT/'runs/configs'/f'{base.name}.json'
    settings=json.loads(config.read_text()) if config.exists() else {}
    return dict(native_source_attempt=base.name,native_allocator_conf=settings.get('allocator_conf','DEFAULT'))
