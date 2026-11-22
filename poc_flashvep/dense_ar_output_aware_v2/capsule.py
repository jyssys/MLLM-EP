"""Version a private worker root without editing the frozen v1 capsule."""
import hashlib
import zipfile
from .common import ROOT,PREVIOUS,write,digest,initialize

def build():
    initialize();target=ROOT/'references/source_capsules/v2_rooted_eager_prefill_native_decode_v1.zip'
    assert not target.exists()
    source=PREVIOUS/'references/source_capsules/eager_prefill_native_decode_overlay_v2.zip'
    assert digest(source)=='2b004c881d5c2f7caf66d6c467957d43079e27a15694cae54b8ee90971504a9f'
    with zipfile.ZipFile(source) as archive:files={n:archive.read(n) for n in archive.namelist()}
    name='poc_flashvep/dense_ar_longbench400_pd/common.py';before=files[name].decode()
    old="ROOT=REPO/'poc_flashvep/reports/dense_ar_longbench400_2d_20261121'"
    assert before.count(old)==1
    files[name]=before.replace(old,"ROOT=REPO/'poc_flashvep/reports/dense_ar_longbench400_v2_20261122'").encode()
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,data)
    record=dict(source=source.name,source_SHA256=digest(source),SHA256=digest(target),changed=['common.ROOT'],
        scope='Only the evidence namespace changed. Fresh v2 runtime and dataset correctness gates remain required.',
        files={n:hashlib.sha256(v).hexdigest() for n,v in files.items()})
    write(target.with_suffix('.json'),record)
    private=ROOT/'references/worker_source_root_v1';private.mkdir()
    with zipfile.ZipFile(target) as archive:archive.extractall(private)
    print('V2_SOURCE_CAPSULE',record['SHA256'],flush=True)

if __name__=='__main__':build()
