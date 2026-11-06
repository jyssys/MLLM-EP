"""Keep the consulted primary-source pages and retrieval outcomes."""
import urllib.request
from .common import *


def main():
    initialize()
    papers = ROOT / 'papers'
    papers.mkdir(exist_ok=True)
    sources = [
        ('load_aware_prefill_deflection','https://arxiv.org/html/2607.02043v1'),
        ('continuous_depth_batching','https://arxiv.org/html/2608.09444v1'),
        ('fluid_pd','https://arxiv.org/html/2610.06917v1'),
        ('exact_loop','https://arxiv.org/html/2611.05150v1')]
    rows=[]
    for name,url in sources:
        target=papers/(name+'.html')
        try:
            if not target.exists():
                with urllib.request.urlopen(url,timeout=30) as stream:
                    target.write_bytes(stream.read())
            rows.append(dict(name=name,url=url,path=str(target.relative_to(ROOT)),SHA256=digest(target),status='RETRIEVED'))
        except Exception as error:
            rows.append(dict(name=name,url=url,status='RETRIEVAL_FAILED',error=repr(error)))
    rows.append(dict(name='Kairos_code_link',url='https://github.com/sudokara/Kairos',
        status='BROWSER_HTTP_404',scope='Paper links this repository. It was not accessible in this audit. No official Kairos code is executed.'))
    rows.append(dict(name='RLT_public_source',url='https://github.com/InferMatrixLab/vllm-rlt',
        status='BROWSER_RETRIEVED',scope='Public Apache-2.0 runtime. Local pinned private code and qualified patches, not current public defaults, determine this experiment.'))
    write(papers/'source_inventory.json',rows)


if __name__=='__main__':
    main()
