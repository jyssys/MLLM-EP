"""Record the installed runtime identity and detect changes from wheel metadata."""
import base64
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import zipfile
from .common import ROOT,write,digest,MODEL


def licenses():
    """Preserve installed license and wheel metadata beside the frozen sources."""
    from .runtime_binding import activate
    activate()
    package=importlib.metadata.distribution('vllm')
    source=ROOT/'references/installed_vllm_python_sources_v1.zip'
    audit=json.loads((ROOT/'references/installed_runtime_source_audit_v1.json').read_text())
    assert package.version==audit['versions']['vllm']
    assert digest(source)==audit['source_archive_SHA256']
    target=ROOT/'references/installed_vllm_source_licenses_metadata_v1.zip'
    manifest=ROOT/'references/installed_vllm_source_licenses_metadata_v1.json'
    if manifest.exists():
        record=json.loads(manifest.read_text());assert digest(target)==record['archive_SHA256']
        return
    assert not target.exists(),'Preserve an incomplete artifact rather than overwrite it'
    selected=[]
    for member in package.files:
        name=str(member)
        is_license=any('license' in p.lower() or p.lower()=='notice' for p in member.parts)
        is_metadata='.dist-info/' in name and member.name in ('METADATA','WHEEL','RECORD')
        if is_license or is_metadata:selected.append(member)
    assert any(member.name=='LICENSE' for member in selected)
    rows=[]
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for member in selected:
            path=package.locate_file(member)
            rows.append(dict(path=str(member),bytes=path.stat().st_size,SHA256=digest(path)))
            archive.write(path,str(member))
    with zipfile.ZipFile(target) as archive:
        for row in rows:
            content=archive.read(row['path'])
            assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['SHA256']
    write(manifest,dict(time=time.time(),archive_SHA256=digest(target),files=rows,
        paired_Python_source_archive_SHA256=digest(source),
        scope='Unmodified installed vLLM license files and wheel metadata accompany the frozen Python source archive. '
            'The two archives retain their installed relative paths. No installed file is edited. '
            'This snapshot does not prove canonical-upstream-wheel identity or reproduce every compiled GPU binary.'))
    print('INSTALLED_RUNTIME_LICENSE_METADATA_SNAPSHOT',len(rows),'files',flush=True)


def main():
    package=importlib.metadata.distribution('vllm');files=[];modified=[]
    archive=ROOT/'references/installed_vllm_python_sources_v1.zip'
    assert not archive.exists(),'Preserve the installed-runtime snapshot'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for member in package.files:
            if str(member).startswith('vllm/') and str(member).endswith('.py'):
                path=package.locate_file(member);sha=digest(path)
                expected=member.hash.value if member.hash and member.hash.mode=='sha256' else None
                match=expected==base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip('=') if expected else None
                row=dict(path=str(member),SHA256=sha,wheel_RECORD_match=match,mtime=path.stat().st_mtime)
                files.append(row)
                if match is False:modified.append(row)
                bundle.write(path,str(member))
    names=('vllm','torch','transformers','nixl','apache-tvm-ffi','nvidia-cutlass-dsl')
    versions={name:importlib.metadata.version(name) for name in names}
    report=dict(time=time.time(),versions=versions,python_files=len(files),modified_from_installed_wheel_RECORD=modified,
        files=files,source_archive_SHA256=digest(archive),model_revision=MODEL.name,
        model_config_SHA256=digest(MODEL/'config.json'),
        scope='Installed-wheel identity and current Python sources. A RECORD match does not prove equivalence to an upstream official wheel. '
            'Custom optional scheduler flags require a separate inactive-default audit. No installed file is modified.')
    write(ROOT/'references/installed_runtime_source_audit_v1.json',report)
    print(json.dumps(dict(versions=versions,python_files=len(files),modified=modified,source_archive_SHA256=digest(archive)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--licenses',action='store_true');args=parser.parse_args()
    licenses() if args.licenses else main()
