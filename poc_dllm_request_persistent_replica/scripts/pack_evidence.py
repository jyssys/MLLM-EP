"""Lossless compact evidence; original capture files are preserved locally.

Repeated empty/position/mask arrays are content-deduplicated, not discarded.
Unpack recovers every original JSON byte and every named NPZ array exactly.
NPZ ZIP-container byte identity is not required; array identity is checked.
"""
import argparse
import gzip
import hashlib
import json
import numpy as np
from poc_dllm_request_persistent_replica.contract import *


def pack_dir(directory,out):
    metadata=dict(source=str(directory),json_files={},npz_files={},original_file_sha256={})
    payload=[];lookup={};offset=0
    for file in sorted(directory.iterdir()):
        if file.suffix=='.json':
            metadata['json_files'][file.name]=file.read_text()
        elif file.suffix=='.npz':
            entries={}
            with np.load(file) as z:
                for key in z.files:
                    a=z[key];data=a.tobytes(order='C');digest=hashlib.sha256(data).hexdigest()
                    identity=(a.dtype.str,a.shape,digest)
                    if identity not in lookup:
                        lookup[identity]=(offset,len(data));payload.append(np.frombuffer(data,dtype=np.uint8));offset+=len(data)
                    start,length=lookup[identity]
                    entries[key]=[start,length,a.dtype.str,list(a.shape),digest]
            metadata['npz_files'][file.name]=entries
        else:continue
        metadata['original_file_sha256'][file.name]=sha(file)
    base=out/directory.name
    if base.with_suffix('.json.gz').exists() or base.with_suffix('.npz').exists():raise FileExistsError(base)
    np.savez_compressed(base.with_suffix('.npz'),payload=np.concatenate(payload) if payload else np.empty(0,dtype=np.uint8))
    with gzip.open(base.with_suffix('.json.gz'),'wt') as f:json.dump(metadata,f,separators=(',',':'))
    verify(base)
    return dict(request=directory.name,original_bytes=sum(f.stat().st_size for f in directory.iterdir()),
        compact_bytes=base.with_suffix('.npz').stat().st_size+base.with_suffix('.json.gz').stat().st_size,
        original_json_files=len(metadata['json_files']),original_npz_files=len(metadata['npz_files']),
        arrays=sum(len(x) for x in metadata['npz_files'].values()),unique_arrays=len(lookup),
        payload_sha256=sha(base.with_suffix('.npz')),metadata_sha256=sha(base.with_suffix('.json.gz')),verified=True)


def verify(base,destination=None):
    with gzip.open(base.with_suffix('.json.gz'),'rt') as f:m=json.load(f)
    with np.load(base.with_suffix('.npz')) as z:payload=z['payload']
    if destination:
        if destination.exists():raise FileExistsError(destination)
        destination.mkdir(parents=True)
    for name,text in m['json_files'].items():
        assert hashlib.sha256(text.encode()).hexdigest()==m['original_file_sha256'][name]
        if destination:(destination/name).write_text(text)
    for name,entries in m['npz_files'].items():
        arrays={}
        for key,(start,length,dtype,shape,digest) in entries.items():
            data=payload[start:start+length]
            assert hashlib.sha256(data.tobytes()).hexdigest()==digest
            arrays[key]=data.view(np.dtype(dtype)).reshape(shape)
        if destination:np.savez_compressed(destination/name,**arrays)


def main():
    p=argparse.ArgumentParser();p.add_argument('--unpack-to');a=p.parse_args()
    if a.unpack_to:
        target=Path(a.unpack_to)
        if target.exists():raise FileExistsError(target)
        source=ROOT/'raw/compact'
        if not source.exists():source=PACKAGE/'raw/compact'
        index=json.loads((source/'index.json').read_text())
        metadata_names=set()
        for record in index['records']:
            name=record['request'];verify(source/name,target/name)
            metadata_names.add(name+'.json.gz')
        for file in sorted(source.glob('*.gz')):
            if file.name not in metadata_names:
                with gzip.open(file,'rb') as f:(target/file.name[:-3]).write_bytes(f.read())
        return
    out=ROOT/'raw/compact'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    records=[]
    for directory in sorted((ROOT/'raw/long_v1').iterdir()):
        if directory.is_dir():records.append(pack_dir(directory,out));print('PACKED',directory.name,flush=True)
    for file in (ROOT/'raw/long_v1').iterdir():
        if file.is_file():
            with gzip.open(out/(file.name+'.gz'),'wb') as f:f.write(file.read_bytes())
    new_json(out/'index.json',dict(records=records,lossless_named_array_and_JSON_byte_parity=True,
        original_capture_preserved=True,original_bytes=sum(r['original_bytes'] for r in records),
        compact_bytes=sum(r['compact_bytes'] for r in records),
        container_caveat='ZIP NPZ file bytes may differ after unpack; every named array and original JSON bytes are identical'))


if __name__=='__main__':main()
