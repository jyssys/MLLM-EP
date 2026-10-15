"""Check frozen references and fixed equivalence controls without model output."""
import sys
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    sys.path.insert(0,str(ASSETS/'cpu_deps'))
    from math_verify import parse,verify,LatexExtractionConfig
    def extract(text):return parse('$'+text+'$',extraction_config=[LatexExtractionConfig()],parsing_timeout=2)
    rows=[]
    for row in read(ROOT/'results/frozen_quality_questions.json'):
        if row['dataset']!='math100':continue
        parsed=extract(row['answer'])
        rows.append(dict(id=row['id'],reference=row['answer'],parsed=bool(parsed),self_equivalent=bool(verify(parsed,parsed,strict=True,timeout_seconds=2))))
    controls=[]
    for gold,response,expected in [(r'\frac{1}{2}','0.5',True),('7','8',False),(r'\sqrt{4}','2',True)]:
        actual=bool(verify(extract(gold),extract(response),strict=True,timeout_seconds=2))
        controls.append(dict(gold=gold,response=response,expected=expected,actual=actual,PASS=actual==expected))
    write(ROOT/'results/public_grader_check.json',dict(references=rows,controls=controls,
        all_pass=all(r['parsed'] and r['self_equivalent'] for r in rows) and all(r['PASS'] for r in controls),
        scope='Frozen reference self-check and fixed parser controls. Not a manual review of generated responses.'))
    assert len(rows)==100 and all(r['parsed'] and r['self_equivalent'] for r in rows)
    assert all(r['PASS'] for r in controls)
    print('PUBLIC_GRADER_CHECK_PASS')


if __name__=='__main__':main()
