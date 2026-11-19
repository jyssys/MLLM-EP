"""Check inherited Python sources against the declared evidence commit."""
import hashlib
import subprocess
import time
import zipfile
from .common import BASE,REPO,ROOT,digest,preserve,write


def main():
    preserve()
    snapshot=ROOT/'raw/calibration_Native_Last0_N36_O16_q0p5_s2026111900_B1_W32_v2/source.zip'
    tree=subprocess.check_output(['git','ls-tree','-rz',BASE],cwd=REPO)
    declared={}
    for entry in tree.split(b'\0'):
        if entry:
            metadata,name=entry.split(b'\t',1)
            declared[name.decode()]=metadata.split()[2].decode()
    rows=[]
    with zipfile.ZipFile(snapshot) as archive:
        for name in archive.namelist():
            if not name.startswith('poc_flashvep/') or not name.endswith('.py') or name.startswith('poc_flashvep/dense_ar_axis_sweetspot/'):
                continue
            data=archive.read(name)
            blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert declared.get(name)==blob,(name,'Snapshot differs from the declared base')
            assert (REPO/name).read_bytes()==data,(name,'Inherited source changed during this campaign')
            rows.append(dict(path=name,Git_blob=blob,SHA256=hashlib.sha256(data).hexdigest()))
    assert rows
    write(ROOT/'analysis/inherited_snapshot_integrity.json',dict(time=time.time(),base=BASE,
        source_snapshot=str(snapshot.relative_to(REPO)),snapshot_SHA256=digest(snapshot),
        inherited_Python_files=len(rows),all_equal_to_base_and_current=True,files=rows,
        scope='The initial immutable worker snapshot and current inherited poc_flashvep Python source equal the declared evidence commit. New campaign files and the generated top-level sitecustomize bootstrap are excluded. This audit does not replace each run source ZIP or the installed-runtime hash checks.'))
    preserve();print('INHERITED_SNAPSHOT_INTEGRITY',len(rows),flush=True)


if __name__=='__main__':
    main()
