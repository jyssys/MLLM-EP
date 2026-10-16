"""Archive version-specific public guidance without changing runtime settings."""
import datetime
import urllib.request
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    entries=[]
    for name,url in [
        ('vllm026_reproducibility','https://docs.vllm.ai/en/v0.26.0/usage/reproducibility/'),
        ('vllm026_batch_invariance','https://docs.vllm.ai/en/v0.26.0/features/batch_invariance/')]:
        target=ROOT/'sources'/f'{name}.html'
        assert not target.exists()
        request=urllib.request.Request(url,headers={'User-Agent':'research-provenance/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response:
            assert response.status==200
            target.write_bytes(response.read())
        entries.append(dict(url=url,local_path=str(target.relative_to(ROOT)),sha256=sha(target),
            bytes=target.stat().st_size,retrieved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    manifest=ROOT/'sources/runtime_docs_manifest.json'
    assert not manifest.exists()
    write(manifest,dict(sources=entries,version='v0.26.0',
        role='Runtime guidance only. No claim that documentation establishes this experiment causal mechanism.'))


if __name__=='__main__':
    main()
