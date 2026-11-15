"""Reuse verified lossless archiving for this campaign's raw data and prompts."""
from .common import *


def selected_files():
    selected=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('worker_source','__pycache__','tools','archives')):continue
        if p.name in ('source.zip','requests.jsonl','offered_requests.jsonl') or p.suffix=='.pt' or p.stat().st_size>=1024**2:
            if p.name!='burner.log':selected.append(p)
    return sorted(selected,key=lambda p:(p.name,str(p)))


def main():
    from ..dense_ar_dynamic_layer_pd import archive_evidence as implementation
    for name in ('ROOT','REPO','SOURCE','BASE','BRANCH','preserve','preservation','digest','write','git','resources'):
        setattr(implementation,name,globals()[name])
    implementation.selected_files=selected_files
    implementation.main()


if __name__=='__main__':main()
