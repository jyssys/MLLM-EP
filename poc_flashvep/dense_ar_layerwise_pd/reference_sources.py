"""Preserve a small, pinned read-only source audit from the official project."""
import json,time,urllib.request
from .common import ROOT,write,digest


def get(url):
    request=urllib.request.Request(url,headers={'User-Agent':'Dense-AR-feasibility-source-audit'})
    with urllib.request.urlopen(request,timeout=30) as response:return response.read()


def main():
    pin='053f80e5201a7c0ab56e468e3d578907a2ca9cc3';repo='scale-snu/layered-prefill'
    directory=ROOT/'references/layered_prefill';directory.mkdir(exist_ok=True)
    tree_url=f'https://api.github.com/repos/{repo}/git/trees/{pin}?recursive=1'
    tree=json.loads(get(tree_url));write(directory/'tree.json',tree)
    endings=('scheduler.py','model_runner.py','sequence.py','qwen3.py','context.py','block_manager.py','llm_engine.py','README.md')
    files=[r['path'] for r in tree['tree'] if r['type']=='blob' and
        (r['path'] in ('README.md','LICENSE') or (r['path'].startswith('nanovllm/') and r['path'].endswith(endings)))]
    manifest=[]
    for path in files:
        url=f'https://raw.githubusercontent.com/{repo}/{pin}/{path}';blob=get(url)
        target=directory/path;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():assert target.read_bytes()==blob
        else:target.write_bytes(blob)
        manifest.append(dict(path=path,url=url,SHA256=digest(target),bytes=len(blob)))
    write(directory/'manifest.json',dict(time=time.time(),repository=repo,commit=pin,files=manifest,
        scope='Pinned official source download for review. It is not an executed baseline or installed runtime.'))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
