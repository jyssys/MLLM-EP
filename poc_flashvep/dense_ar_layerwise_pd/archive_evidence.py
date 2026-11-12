"""Preserve large evidence in verified lossless parts below GitHub limits."""
import hashlib,json,subprocess,tarfile
from .common import *


def selected_files():
    values=[]
    for p in (ROOT/'raw').rglob('*'):
        if not p.is_file() or 'worker_source' in p.parts or '__pycache__' in p.parts:continue
        if p.suffix=='.pt' or p.stat().st_size>=48*1024**2:values.append(p)
    # Place the same prompt near its repeated reference and candidate states.
    return sorted(values,key=lambda p:(p.name.rsplit('_',1)[-1],p.name,str(p)))


def main():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists(),'Archive only after GPU work closes'
    owner=resources();active=owner.processes();burners=owner.burners()
    assert all(os.getsid(x['pid'])==burners.get(x['gpu']) for x in active),'Archive outside GPU measurements'
    folder=ROOT/'archives';folder.mkdir(exist_ok=True);manifest=folder/'large_evidence_manifest.json'
    assert not manifest.exists(),'Do not overwrite a frozen archive'
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
            for b in iter(lambda:f.read(1024**2),b''):h.update(b)
            checked.append(dict(path=member.name,bytes=member.size,SHA256=h.hexdigest()))
    assert decoder.wait()==0 and checked==rows,'Lossless archive verification failed'
    parts=[]
    with archive.open('rb') as source:
        index=0
        while data:=source.read(48*1024**2):
            p=folder/f'large_evidence.tar.zst.part{index:04d}'
            with p.open('wb') as out:out.write(data)
            parts.append(dict(path=str(p.relative_to(REPO)),bytes=len(data),SHA256=digest(p)));index+=1
    write(manifest,dict(original_files=rows,parts=parts,compressed_SHA256=digest(archive),
        original_bytes=sum(x['bytes'] for x in rows),compressed_bytes=archive.stat().st_size,
        verified_lossless=True,originals_preserved_locally=True,
        restore='Concatenate parts in numeric order. Decompress with zstd --long=31. Extract the tar at the repository root. Check each original SHA256.',
        scope='Large binary evidence only. The local unsplit archive and original files stay on disk. Worker source ZIP archives separately preserve extracted worker_source trees. No original evidence is removed.'))
    print(json.dumps(dict(original_bytes=sum(x['bytes'] for x in rows),compressed_bytes=archive.stat().st_size,parts=len(parts))))


if __name__=='__main__':main()
