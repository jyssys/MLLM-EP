"""Compare predecessor sources with the declared publication parent."""
import json
import subprocess
import time
from .common import ROOT,REPO,BASE,PACKAGES,initialize,write,digest


def main():
    initialize()
    paths=[p for name in PACKAGES for p in (REPO/'poc_flashvep'/name).glob('*.py')]
    names=[str(p.relative_to(REPO)) for p in paths]
    actual=subprocess.check_output(['git','hash-object','--stdin-paths'],cwd=REPO,
        input=('\n'.join(names)+'\n').encode()).decode().splitlines()
    tree=subprocess.check_output(['git','ls-tree','-r',BASE,'--',
        *['poc_flashvep/'+name for name in PACKAGES]],cwd=REPO,text=True)
    expected={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
    rows=[dict(path=name,current_blob_sha=sha,base_blob_sha=expected.get(name),
        equal_to_base=expected.get(name)==sha,SHA256=digest(path)) for name,sha,path in zip(names,actual,paths)]
    write(ROOT/'state/predecessor_vs_228a_source_audit.json',dict(time=time.time(),base=BASE,
        files=len(rows),equal=sum(r['equal_to_base'] for r in rows),
        changed=[r for r in rows if not r['equal_to_base']],rows=rows,
        scope='Read-only source audit. Preserve every current file. The worker ZIP records the actual execution source, including new campaign adapters.'))
    write(ROOT/'references/source_review.json',dict(time=time.time(),
        historical_reports=[
            'dense_ar_layer_token_dynamic_headroom_final_summary.md',
            'dense_ar_layer_vs_token_online_results.md',
            'dense_ar_layer_vs_token_correctness_and_execution.md',
            'duetserve_agg_dp2_tp2_tbt_crossover_final_summary.md'],
        Notion=dict(URL='https://app.notion.com/p/4170ee8277e481e9848ac7283309f7df',
            inspected=True,page_last_edited_at='2026-11-16T16:24:44.170Z',
            scope='Historical checklist. Quantitative claims are checked against repository reports and raw evidence.'),
        DuetServe=dict(URL='https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf',
            inspected=True,scope='Figure2 motivates the fixed DP2 workload. It is not evidence of our new PD performance.')))
    print('SOURCE_AUDIT',len(rows),sum(r['equal_to_base'] for r in rows))


if __name__=='__main__':
    main()
