"""Copy approved runtime source text. Never edit installed files."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
from .common import ROOT,write,digest,preserve


def main():
    preserve()
    declared=json.loads((ROOT/'state/installed_runtime_sources.json').read_text())['hashes']
    rows=[]
    for filename,expected in declared.items():
        source=Path(filename)
        assert source.suffix=='.py' and digest(source)==expected
        relative=Path(str(source).split('/site-packages/',1)[1])
        target=ROOT/'references/runtime_source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            assert digest(target)==expected
        else:
            shutil.copyfile(source,target)
        rows.append(dict(original=str(source),snapshot=str(target.relative_to(ROOT)),SHA256=digest(target)))
    licenses=[]
    distribution=importlib.metadata.distribution('vllm')
    for path in distribution.files or []:
        if path.name.upper() not in ('LICENSE','LICENSE.TXT','NOTICE','NOTICE.TXT'):
            continue
        source=Path(distribution.locate_file(path))
        if not source.is_file() or source.stat().st_size>1024**2:
            continue
        # Distinct third-party directories can each contain a different LICENSE.
        target=ROOT/'references/runtime_source_snapshot/licenses'/(digest(source)[:16]+'_'+path.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copyfile(source,target)
        assert digest(source)==digest(target)
        licenses.append(dict(source=str(source),snapshot=str(target.relative_to(ROOT)),SHA256=digest(target)))
    write(ROOT/'references/runtime_source_snapshot_manifest.json',dict(time=time.time(),sources=rows,licenses=licenses,
        scope='Exact source text of the approved installed vLLM paths. External files stay unchanged. These snapshots do not include binary dependencies, model weights or a complete deployment image.'))
    preserve();print('INSTALLED_RUNTIME_TEXT_CAPTURED',len(rows),len(licenses),flush=True)


if __name__=='__main__':
    main()
