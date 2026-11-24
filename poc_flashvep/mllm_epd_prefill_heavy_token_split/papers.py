"""Preserve primary papers and extract text for the scoped novelty audit."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import time

sys.path.insert(0,'/home/esjung/external/mllm-epd-prefill-heavy-deps-20261124')
from .common import ROOT,write,digest

SOURCES={
    'epd_icml2025':'https://raw.githubusercontent.com/mlresearch/v267/main/assets/singh25d/singh25d.pdf',
    'modserve':'https://arxiv.org/pdf/2502.00937v3',
    'epd_serve_ascend':'https://arxiv.org/pdf/2601.11590v1',
    'duetserve':'https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf',
    'fluidpd':'https://arxiv.org/pdf/2610.06917v1',
    'kairos':'https://arxiv.org/pdf/2607.02043v1',
    'raincheck_withdrawn_v1':'https://arxiv.org/pdf/2610.39562v1',
    'easerve':'https://arxiv.org/pdf/2609.31551v1',
    'hydrainfer':'https://arxiv.org/pdf/2505.12658v1',
}


def fetch(pair):
    import httpx
    import fitz
    name,url=pair;folder=ROOT/'references/primary_papers';folder.mkdir(exist_ok=True)
    result=dict(name=name,url=url,time=time.time())
    try:
        p=folder/(name+'.pdf')
        if not p.exists():
            r=httpx.get(url,follow_redirects=True,timeout=120);r.raise_for_status()
            assert r.content.startswith(b'%PDF'),(r.status_code,r.headers.get('content-type'))
            p.write_bytes(r.content)
        doc=fitz.open(p);text='\n'.join('PAGE '+str(i+1)+'\n'+page.get_text() for i,page in enumerate(doc))
        target=folder/(name+'.txt');target.write_text(text)
        result.update(passed=True,pages=len(doc),PDF_SHA256=digest(p),text_SHA256=digest(target))
    except Exception as error:result.update(passed=False,error=type(error).__name__+': '+str(error))
    write(folder/(name+'.json'),result);return result


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(fetch,SOURCES.items()))
    write(ROOT/'references/primary_paper_fetch.json',dict(time=time.time(),sources=results))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
