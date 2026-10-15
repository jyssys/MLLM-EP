"""Preserve large evidence outside Git. Keep original evidence unchanged."""
import argparse
import shutil
from poc_looped_recurrence_tp_4models.contract import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore',action='store_true');args=parser.parse_args()
    manifest=ROOT/'results/heavy_evidence_manifest.json'
    if args.restore:
        rows=json.loads(manifest.read_text())['files']
        for row in rows:
            target=ROOT/row['relative_path'];source=Path(row['external_path'])
            if target.exists():assert sha(target)==row['sha256'];continue
            assert source.exists() and sha(source)==row['sha256']
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        print('Restored or checked',len(rows),'evidence files');return
    rows=[]
    for source in sorted((ROOT/'results').rglob('*')):
        if not source.is_file():continue
        if not (source.name.startswith('torch_events') and source.name.endswith('.json.gz') or
                source.name.startswith('parity') and source.suffix=='.npz' or
                source.name=='parity_natural_selected.json.gz'):continue
        relative=source.relative_to(ROOT);target=ASSETS/'heavy'/relative
        digest=sha(source)
        if target.exists():assert sha(target)==digest,'Never overwrite a different archived artifact'
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        rows.append(dict(relative_path=str(relative),external_path=str(target),bytes=source.stat().st_size,sha256=digest))
    write(manifest,dict(files=rows,total_bytes=sum(r['bytes'] for r in rows),
        scope='Full parity arrays and profiler events. Originals remain unchanged. Git contains compact checks and a hash manifest.'))
    print('Preserved',len(rows),'files;',sum(r['bytes'] for r in rows)/2**20,'MiB outside Git')


if __name__=='__main__':main()
