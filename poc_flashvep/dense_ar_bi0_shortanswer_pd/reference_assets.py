"""Save primary source text and a pinned local-only scoring dependency."""
import hashlib
import json
import subprocess
import urllib.request
from .common import ROOT,PYTHON,write,digest


def licenses():
    sources=dict(
        LongBench_LICENSE='https://raw.githubusercontent.com/THUDM/LongBench/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LICENSE',
        NIXL_LICENSE='https://raw.githubusercontent.com/ai-dynamo/nixl/v1.4.1/LICENSE')
    rows=[]
    for name,url in sources.items():
        path=ROOT/'references'/(name+'.txt')
        if not path.exists():
            with urllib.request.urlopen(url,timeout=40) as response:path.write_bytes(response.read())
        rows.append(dict(name=name,URL=url,SHA256=digest(path)))
    write(ROOT/'references/third_party_license_manifest.json',dict(sources=rows,
        scope='Unmodified license notices for the pinned software reference files. Dataset source provenance is separate. '
        'This manifest does not replace the terms of the underlying task datasets or the model.'))


def main():
    urls = dict(
        LongBench_metrics='https://raw.githubusercontent.com/THUDM/LongBench/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py',
        LongBench_eval='https://raw.githubusercontent.com/THUDM/LongBench/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/eval.py',
        NIXL_UCX_backend='https://raw.githubusercontent.com/ai-dynamo/nixl/v1.4.1/src/plugins/ucx/ucx_backend.cpp')
    rows = []
    for name,url in urls.items():
        path = ROOT/'references'/(name+'.txt')
        if not path.exists():
            with urllib.request.urlopen(url,timeout=40) as response:
                content=response.read().decode()
            path.write_text(content)
        rows.append(dict(name=name,url=url,SHA256=digest(path)))
    with urllib.request.urlopen('https://pypi.org/pypi/rouge/1.0.1/json',timeout=40) as response:
        metadata=json.loads(response.read())
    wheel=next(r for r in metadata['urls'] if r['filename'].endswith('py3-none-any.whl'))
    target=ROOT/'references'/wheel['filename']
    if not target.exists():
        with urllib.request.urlopen(wheel['url'],timeout=40) as response:
            target.write_bytes(response.read())
    assert digest(target)==wheel['digests']['sha256']
    deps=ROOT/'tools/quality_deps'
    if not (deps/'rouge').exists():
        subprocess.run([str(PYTHON),'-m','pip','install','--no-deps','--target',str(deps),str(target)],check=True)
    write(ROOT/'references/primary_source_manifest.json',dict(sources=rows,
        rouge_wheel=dict(version='1.0.1',URL=wheel['url'],SHA256=digest(target)),
        scope='Read-only primary sources. The scoring dependency is installed only in this campaign tools directory. No serving runtime package is changed.',
        Notion_archive=dict(URL='https://app.notion.com/p/1062a9a81d8641f895f0bdf6c0702489',
            coverage='Database description covers183 entries through2026-11-10. It does not supersede the newer Git evidence. No Notion write.')))


if __name__=='__main__':
    main()
