"""Check preserved sources and actual core overlay versions."""
import hashlib
import json
import zipfile
from pathlib import Path
from .common import ROOT,NUMERICAL,REPO,digest,write


def main():
    lock=json.loads((NUMERICAL/'configs/read_only_source_lock.json').read_text())
    differences=[path for path,sha in lock.items() if not Path(path).exists() or digest(path)!=sha]
    versions={}
    for folder in sorted((ROOT/'raw').glob('map_*')):
        archive=folder/'adaptive_source.zip'
        if not archive.exists():continue
        with zipfile.ZipFile(archive) as z:
            versions[folder.name]={name:hashlib.sha256(z.read('poc_flashvep/adaptive_recurrence_cut/'+name)).hexdigest()
                                   for name in ('staged.py','connector.py')}
    current={name:digest(REPO/'poc_flashvep/adaptive_recurrence_cut'/name) for name in ('staged.py','connector.py')}
    matches={name:value==current for name,value in versions.items()}
    result=dict(preserved_source_files=len(lock),source_changes=differences,current_core_overlay_SHA256=current,
                map_core_overlay_checks=matches,passed=not differences and all(matches.values()),
                scope='Model execution, retained state, and cut connector. Worker harness changes remain archived per run.')
    write(ROOT/'state/final_source_preservation.json',result);print({k:v for k,v in result.items() if k!='map_core_overlay_checks'})
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
