"""Verify installed wheel sources without editing the installed runtime."""
import base64
import hashlib
import importlib.metadata
import json
import time
from urllib.parse import urlsplit,urlunsplit
from .common import ROOT,write


def main():
    distribution=importlib.metadata.distribution('vllm');rows=[]
    for entry in distribution.files:
        if not str(entry).startswith('vllm/') or not str(entry).endswith('.py') or not entry.hash:continue
        path=distribution.locate_file(entry)
        actual=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=') if path.exists() else None
        rows.append(dict(path=str(path),RECORD_hash=entry.hash.value,actual_RECORD_encoding=actual,
            passed=entry.hash.mode=='sha256' and actual==entry.hash.value))
    direct=distribution.read_text('direct_url.json');origin=json.loads(direct) if direct else None
    if origin and origin.get('url'):
        parts=urlsplit(origin['url']);origin['url']=urlunsplit((parts.scheme,parts.hostname or '',parts.path,'',''))
    versions={name:importlib.metadata.version(name) for name in ('vllm','torch','transformers')}
    try:versions['nixl']=importlib.metadata.version('nixl-cu12')
    except importlib.metadata.PackageNotFoundError:versions['nixl']=importlib.metadata.version('nixl')
    result=dict(time=time.time(),versions=versions,checked_python_sources=len(rows),files=rows,
        mismatches=[r for r in rows if not r['passed']],passed=bool(rows) and all(r['passed'] for r in rows),
        wheel_origin=origin,evidence='INSTALLED_WHEEL_RECORD_VERIFICATION',
        scope='Checks installed Python files against this wheel RECORD. No package file is changed. '
        'This does not independently certify every upstream build input, GPU binary, driver or performance property.')
    write(ROOT/'state/installed_vllm_source_integrity.json',result);assert result['passed']
    print(versions,len(rows),result['passed'])


if __name__=='__main__':main()
