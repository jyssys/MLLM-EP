"""Check report links, source preservation and frozen candidate inputs."""
import json
import re
from .common import ROOT,REPO,SOURCE,digest,write,preserve
from .publication import DOCUMENTS


def main():
    preserve();missing=[];files=[]
    for name in DOCUMENTS:
        path=REPO/'poc_flashvep/reports'/name
        assert path.exists(),name
        prose=path.read_text();files.append(dict(path=str(path.relative_to(REPO)),SHA256=digest(path)))
        for target in re.findall(r'\]\(([^)]+)\)',prose):
            target=target.strip('<>').split('#',1)[0]
            if not target or '://' in target or target.startswith('mailto:'):
                continue
            resolved=(path.parent/target).resolve()
            if not resolved.exists():
                missing.append(dict(report=name,target=target))
        assert 'TODO' not in prose and 'TBD' not in prose,name
    assert not missing,missing
    frozen=[]
    for path in sorted((ROOT/'state').glob('candidate_confirmation_freeze_O*.json')):
        contract=json.loads(path.read_text())['contract']
        checks={name:digest(SOURCE/name)==expected for name,expected in contract['worker_hashes'].items()}
        checks.update(SLO=digest(ROOT/'configs/frozen_SLO_regimes.json')==contract['SLO_SHA256'],
            dataset=digest(ROOT/'data/heldout128_8000.json')==contract['dataset_SHA256'])
        assert all(checks.values()),checks
        frozen.append(dict(freeze=path.name,checks=checks))
    write(ROOT/'analysis/report_and_freeze_validation.json',dict(passed=True,reports=files,
        missing_links=missing,frozen_candidate_inputs=frozen,
        scope='Local evidence paths must exist before archiving. Large members can require archive restoration in a fresh clone. This check does not turn a numerical or performance failure into a pass.'))
    print('REPORT_AND_FREEZE_CHECKS_PASS',len(files),len(frozen),flush=True)


if __name__=='__main__':
    main()
