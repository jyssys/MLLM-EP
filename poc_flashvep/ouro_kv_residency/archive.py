"""Preserve immutable raw files in portable, size-bounded Git archives."""
import json
import tarfile
import shutil
from collections import defaultdict
from .common import ROOT,REPO,FIXTURES,digest,write

def main():
    archives=ROOT/'archives';archives.mkdir(exist_ok=True)
    # Preserve the initial gzip attempt outside the Git delivery set.
    legacy=ROOT/'state'/'initial_gzip_archives'
    for path in archives.glob('*.tar.gz'):
        legacy.mkdir(exist_ok=True)
        target=legacy/path.name
        assert not target.exists(),target
        shutil.move(path,target)
    groups=defaultdict(list)
    for path in sorted((ROOT/'raw').glob('*')):
        if not path.is_dir():continue
        if path.name.startswith('grid'):group='raw_grid'
        elif path.name.startswith('cal_'):group='raw_calibration'
        elif path.name.startswith('layout_'):group='raw_layout'
        elif path.name.startswith(('fixture','correct_')):group='raw_fixtures'
        else:group='raw_controls'
        groups[group].extend(p for p in path.rglob('*') if p.is_file())
    for path in sorted((ROOT/'distributed').glob('*')):
        if path.is_dir():groups[f'distributed_{path.name}']=sorted(p for p in path.rglob('*') if p.is_file())
    for prefix in ('profiles','logs','counters'):
        groups[prefix]=sorted(p for p in (ROOT/prefix).rglob('*') if p.is_file())
    manifest=[];originals=[]
    for name,paths in groups.items():
        output=archives/f'{name}.tar.xz'
        with tarfile.open(output,'w:xz',preset=3) as archive:
            for path in paths:
                relative=str(path.relative_to(ROOT));archive.add(path,arcname=relative,recursive=False)
                originals.append(dict(path=relative,sha256=digest(path),bytes=path.stat().st_size,archive=output.name))
        assert output.stat().st_size<95*1024**2,('ARCHIVE_TOO_LARGE_FOR_GIT',output)
        manifest.append(dict(path=str(output.relative_to(ROOT)),bytes=output.stat().st_size,sha256=digest(output)))
    spec=REPO/'poc_flashvep/reports/ouro_loop_aware_kv_residency_distributed_attention_poc_spec.md'
    output=archives/'authoritative_contract.tar.xz'
    with tarfile.open(output,'w:xz',preset=3) as archive:archive.add(spec,arcname='contract/authoritative_spec.md',recursive=False)
    manifest.append(dict(path=str(output.relative_to(ROOT)),bytes=output.stat().st_size,sha256=digest(output)))
    write(ROOT/'raw_archive_manifest.json',dict(archives=manifest,original_files=originals,
        authoritative_spec=dict(path=str(spec),sha256=digest(spec)),
        tensor_fixtures=dict(path=str(FIXTURES),policy='Retain locally. Capture manifests preserve full hashes. Reconstruct with the pinned model and inputs.')))
    print('ARCHIVES',len(manifest),'RAW_FILES',len(originals),'COMPRESSED_BYTES',sum(r['bytes'] for r in manifest))

if __name__=='__main__':main()
