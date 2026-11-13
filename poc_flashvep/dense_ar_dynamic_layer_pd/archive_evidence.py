"""Preserve repeated raw evidence in verified lossless archive parts."""
import hashlib,subprocess,tarfile
from .common import *


def selected_files():
    selected=[]
    for p in (ROOT/'raw').rglob('*'):
        if not p.is_file() or 'worker_source' in p.parts or '__pycache__' in p.parts:continue
        if p.name in ('source.zip','requests.jsonl','offered_requests.jsonl') or p.suffix=='.pt' or p.stat().st_size>=1024**2:
            selected.append(p)
    return sorted(selected,key=lambda p:(p.name,str(p)))


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'Close GPU work before archiving'
    owner=resources();active=owner.processes();burners=owner.burners()
    assert all(os.getsid(r['pid'])==burners.get(r['gpu']) for r in active)
    folder=ROOT/'archives';folder.mkdir(exist_ok=True);manifest=folder/'large_evidence_manifest.json'
    assert not manifest.exists(),'Preserve any existing frozen archive'
    paths=selected_files();rows=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p)) for p in paths]
    names=folder/'large_evidence_files.null';names.write_bytes(b''.join(str(p.relative_to(REPO)).encode()+b'\0' for p in paths))
    archive=folder/'large_evidence.tar.zst.local'
    with names.open('rb') as source:
        tar=subprocess.Popen(['tar','--null','-T','-','-cf','-'],cwd=REPO,stdin=source,stdout=subprocess.PIPE)
        zipper=subprocess.Popen(['/home/esjung/anaconda3/bin/zstd','-3','--long=31','-T1','-o',str(archive)],stdin=tar.stdout)
        tar.stdout.close();assert zipper.wait()==0 and tar.wait()==0
    decoder=subprocess.Popen(['/home/esjung/anaconda3/bin/zstd','-dc','--long=31',str(archive)],stdout=subprocess.PIPE)
    checked=[]
    with tarfile.open(fileobj=decoder.stdout,mode='r|') as stream:
        for member in stream:
            if not member.isfile():continue
            h=hashlib.sha256();f=stream.extractfile(member)
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
            checked.append(dict(path=member.name,bytes=member.size,SHA256=h.hexdigest()))
    assert decoder.wait()==0 and checked==rows,'Archive verification failed'
    parts=[]
    with archive.open('rb') as stream:
        i=0
        while data:=stream.read(48*1024**2):
            path=folder/f'large_evidence.tar.zst.part{i:04d}';path.write_bytes(data)
            parts.append(dict(path=str(path.relative_to(REPO)),bytes=len(data),SHA256=digest(path)));i+=1
    write(manifest,dict(original_files=rows,parts=parts,compressed_SHA256=digest(archive),
        original_bytes=sum(r['bytes'] for r in rows),compressed_bytes=archive.stat().st_size,
        verified_lossless=True,originals_preserved_locally=True,
        restore='Concatenate numbered parts. Decompress with zstd --long=31. Extract the tar at the repository root. Check each original SHA256.',
        scope='All original evidence remains local. The archive includes repeated prompts, requests, and per-job frozen source ZIP files. Nothing is deleted.'))
    print('VERIFIED_ARCHIVE',sum(r['bytes'] for r in rows),archive.stat().st_size,len(parts))


if __name__=='__main__':main()
