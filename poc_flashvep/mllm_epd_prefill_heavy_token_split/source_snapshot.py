"""Archive current source bytes and check them against recorded boot hashes."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import time
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

from .common import ROOT, SOURCE, REPO, digest, preserve, write


def snapshot(label, include_runtime=False):
    assert re.fullmatch('[A-Za-z0-9_.-]+',label) and label not in ('.','..')
    preserve()
    paths={str(p.relative_to(REPO)):p for p in SOURCE.glob('*.py')}
    for package in ['dense_ar_slo_layer_pd','ouro_budget_joint_cg','dense_ar_bi0_axis_rootcause']:
        folder=REPO/'poc_flashvep'/package
        paths.update({str(p.relative_to(REPO)):p for p in folder.glob('*.py')})
    if include_runtime:
        spec=importlib.util.find_spec('vllm')
        runtime=Path(spec.origin).parent
        paths.update({'installed_vllm/'+str(p.relative_to(runtime)):p for p in runtime.rglob('*.py')})
    records=[]
    name=f'{label}_{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    archive=ROOT/'archives'/f'{name}.source.zip'
    archive.parent.mkdir(parents=True,exist_ok=True)
    hashes={}
    with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=6) as target:
        for key,path in sorted(paths.items()):
            content=path.read_bytes()
            sha=hashlib.sha256(content).hexdigest()
            assert digest(path)==sha, 'Source changed during archival: '+str(path)
            target.writestr(key,content)
            records.append(dict(member=key,original=str(path),bytes=len(content),SHA256=sha))
            if path.parent==SOURCE:hashes[path.name]=sha
        target.writestr('source_manifest.json',json.dumps(records,indent=2)+'\n')
    with ZipFile(archive) as target:
        assert target.testzip() is None
        assert all(hashlib.sha256(target.read(r['member'])).hexdigest()==r['SHA256'] for r in records)
    leases=[]
    for path in sorted((ROOT/'state').glob('server_*.json')):
        lease=json.loads(path.read_text())
        boot=lease.get('source_SHA256',{})
        leases.append(dict(name=lease.get('name'),finished=lease.get('finished'),
            matching_boot_members=[name for name,sha in boot.items() if hashes.get(name)==sha],
            unavailable_or_changed_boot_members=[name for name,sha in boot.items() if hashes.get(name)!=sha]))
    record=dict(time=time.time(),archive=str(archive),archive_SHA256=digest(archive),
        files=len(records),source_member_SHA256=hashes,boot_hash_comparison=leases,
        runtime_included=include_runtime,checksum_verification='PASS',
        scope='Current on-disk Python source. A match to a recorded boot hash preserves those exact bytes. '
            'Changed or missing boot bytes are not reconstructed. Python source does not include native libraries or model weights.')
    write(ROOT/'state'/f'{name}.source_snapshot.json',record)
    preserve()
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--label',required=True)
    parser.add_argument('--include-runtime',action='store_true')
    args=parser.parse_args()
    result=snapshot(args.label,args.include_runtime)
    print(json.dumps({key:result[key] for key in ['archive','archive_SHA256','files','runtime_included','checksum_verification']}))
