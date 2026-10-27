"""Store complete evidence in bounded archive parts. Do not remove source files."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import time
from .common import ROOT,setup,write,digest

def archive(name,targets):
    index=ROOT/'archives'/f'{name}.index.json'
    if index.exists():
        saved=json.loads(index.read_text())
        for part in saved['parts']:
            assert digest(ROOT/'archives'/part['name'])==part['sha256']
        print('VERIFIED_EXISTING_ARCHIVE',name,flush=True)
        return
    sources=[ROOT/path for path in targets]
    assert all(path.exists() for path in sources)
    cap=80_000_000
    with tempfile.TemporaryDirectory(prefix='refinement-ep-evidence-') as temporary:
        packed=Path(temporary)/f'{name}.tar.gz'
        with tarfile.open(packed,'w:gz',compresslevel=1) as output:
            for path in sources:
                output.add(path,arcname=str(path.relative_to(ROOT)))
        checksum=digest(packed)
        parts=[]
        with packed.open('rb') as stream:
            number=0
            while block:=stream.read(cap):
                path=ROOT/'archives'/f'{name}.tar.gz.part{number:03d}'
                h=hashlib.sha256(block).hexdigest()
                if path.exists():
                    assert digest(path)==h,'Existing archive part differs. Preserve it for review.'
                else:
                    with path.open('xb') as out:
                        out.write(block)
                parts.append(dict(name=path.name,bytes=len(block),sha256=h))
                number+=1
        assert sum(p['bytes'] for p in parts)==packed.stat().st_size
        write(index,dict(name=name,time=time.time(),source_paths=targets,parts=parts,
            archive_sha256=checksum,archive_bytes=packed.stat().st_size,
            format='gzip-compressed tar, split at 80,000,000 bytes',
            restore='Concatenate parts in numeric order, verify the archive SHA-256, and extract into a new directory.',
            original_raw_files='PRESERVED_UNCHANGED'))
        print('ARCHIVED',name,'BYTES',packed.stat().st_size,'PARTS',len(parts),flush=True)

def main():
    setup()
    archive('interrupted_full_gsm8k',[
        'ep4/A_full_gsm8k','hf/full_gsm8k','logs/A_full_gsm8k.log',
        'logs/hf_gpu4_gsm8k.log','logs/hf_gpu5_gsm8k.log','logs/hf_gpu6_gsm8k.log',
        'logs/hf_gpu7_gsm8k.log','pilot/prior_full_controller_state.json'])
    archive('retained_all_validation_traces',[
        'ep4/A_pilot_validation_trace','ep4/B_pilot_validation_trace',
        'logs/A_pilot_validation_trace.log','logs/B_pilot_validation_trace.log',
        'state/pilot_traces.json'])

if __name__=='__main__':
    main()
