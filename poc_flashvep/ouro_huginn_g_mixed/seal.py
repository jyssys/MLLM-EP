"""Hash only this campaign's portable code and evidence artifacts."""
import json
from pathlib import Path
from .common import ROOT,REPO,digest,write

def main():
    paths=[]
    for base in [Path(__file__).parent,ROOT]:
        for path in base.rglob('*'):
            if not path.is_file():continue
            rel=path.relative_to(base)
            if any(p in ['__pycache__','raw','state','logs','profiles'] for p in rel.parts):continue
            if path.name in ['artifact_manifest.json','burner.log','research_state.json','batch_plan_trace.csv','all_ITL_gaps.csv']:continue
            paths.append(path)
    rows=[dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)]
    write(ROOT/'artifact_manifest.json',dict(artifacts=rows,scope='Only this PoC code and portable evidence. No model weights.'))
    print('SEALED',len(rows),'portable artifacts')

if __name__=='__main__':main()
