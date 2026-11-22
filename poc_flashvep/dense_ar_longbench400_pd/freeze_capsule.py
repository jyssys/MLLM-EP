"""Freeze one immutable source archive for independent confirmation starts."""
import json
import fcntl
from pathlib import Path
import zipfile
from .common import ROOT,REPO,write,digest


def freeze(label):
    assert label and all(c.isalnum() or c in '_-' for c in label)
    with (ROOT/'state'/('capsule_'+label+'.lock')).open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        return _freeze(label)


def _freeze(label):
    path=ROOT/'references/source_capsules'/(label+'.zip')
    manifest=path.with_suffix('.json')
    if path.exists():
        proof=json.loads(manifest.read_text());assert digest(path)==proof['SHA256'];return path.name,proof['SHA256']
    packages=('dense_ar_longbench400_pd','dense_ar_bi0_shortanswer_pd','dense_ar_slo_layer_pd')
    files=[REPO/'poc_flashvep/__init__.py']
    for package in packages:files.extend(p for p in (REPO/'poc_flashvep'/package).glob('*') if p.suffix in ('.py','.cc','.cu','.h'))
    hashes={str(p.relative_to(REPO)):digest(p) for p in sorted(files)}
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(hashes):
            entry=zipfile.ZipInfo(item,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,(REPO/item).read_bytes())
    value=dict(SHA256=digest(path),files=hashes,
        scope='One immutable source capsule for all named confirmation arms. Later development does not change it. '
            'The installed runtime and model have separate full-file audits.')
    write(manifest,value);return path.name,value['SHA256']


if __name__=='__main__':
    import sys
    print(freeze(sys.argv[1]),flush=True)
