"""Preserve large diagnostic analysis files in verified archive parts."""
import hashlib
import json
import argparse
import subprocess
import tarfile
from .common import REPO, ROOT, digest, preserve, write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--remaining-raw',action='store_true')
    args=parser.parse_args()
    preserve()
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    folder = ROOT/'archives'/('raw_metadata_addendum' if args.remaining_raw else 'analysis_addendum')
    folder.mkdir(exist_ok=True)
    manifest = folder/'large_evidence_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        assert saved['verified_lossless']
        assert all(digest(REPO/r['path'])==r['SHA256'] for r in saved['original_files'])
        assert all(digest(REPO/r['path'])==r['SHA256'] for r in saved['parts'])
        print('DIAGNOSTIC_ANALYSIS_ARCHIVE_ALREADY_VERIFIED', len(saved['original_files']), flush=True)
        return
    if args.remaining_raw:
        earlier=json.loads((ROOT/'archives/large_evidence_manifest.json').read_text())
        archived={r['path'] for r in earlier['original_files']}
        direct={'configuration.json','command.json','dispatcher_result.json','window.json',
                'input_provenance.json','unified_worker_gate.json'}
        paths=sorted(p for p in (ROOT/'raw').rglob('*') if p.is_file()
            and 'worker_source' not in p.parts and '__pycache__' not in p.parts
            and p.suffix!='.bin' and p.name not in direct and str(p.relative_to(REPO)) not in archived)
    else:
        paths = sorted((ROOT/'analysis').glob('*kernel_intervals.json'))
    assert paths
    rows = [dict(path=str(p.relative_to(REPO)),bytes=p.stat().st_size,SHA256=digest(p)) for p in paths]
    stem='raw_metadata' if args.remaining_raw else 'diagnostic_analysis'
    archive = folder/(stem+'.tar.zst.local')
    assert not archive.exists(), 'Preserve an interrupted archive and use a new namespace'
    payload = b''.join(str(p.relative_to(REPO)).encode()+b'\0' for p in paths)
    names = folder/(stem+'_files.null')
    names.write_bytes(payload)
    with names.open('rb') as source:
        tar = subprocess.Popen(['tar','--null','-T','-','-cf','-'],cwd=REPO,stdin=source,stdout=subprocess.PIPE)
        zipper = subprocess.Popen(['/home/esjung/anaconda3/bin/zstd','-3','--long=31','-T1','-o',str(archive)],stdin=tar.stdout)
        tar.stdout.close()
        assert zipper.wait()==0 and tar.wait()==0
    decoder = subprocess.Popen(['/home/esjung/anaconda3/bin/zstd','-dc','--long=31',str(archive)],stdout=subprocess.PIPE)
    checked = []
    with tarfile.open(fileobj=decoder.stdout,mode='r|') as stream:
        for member in stream:
            if not member.isfile():
                continue
            h=hashlib.sha256()
            source=stream.extractfile(member)
            for block in iter(lambda:source.read(1024**2),b''):
                h.update(block)
            checked.append(dict(path=member.name,bytes=member.size,SHA256=h.hexdigest()))
    assert decoder.wait()==0 and checked==rows
    parts=[]
    with archive.open('rb') as stream:
        index=0
        while payload:=stream.read(48*1024**2):
            path=folder/f'{stem}.tar.zst.part{index:04d}'
            path.write_bytes(payload)
            parts.append(dict(path=str(path.relative_to(REPO)),bytes=len(payload),SHA256=digest(path)))
            index+=1
    write(manifest,dict(original_files=rows,parts=parts,
        compressed_SHA256=digest(archive),original_bytes=sum(r['bytes'] for r in rows),
        compressed_bytes=archive.stat().st_size,verified_lossless=True,originals_preserved_locally=True,
        restore='Concatenate numbered parts. Decompress with zstd --long=31. Extract at the repository root. Verify each original SHA256.',
        scope=('Remaining immutable raw metadata and traces. Keep original files locally. Core configuration, command, receipt, window, provenance and worker gates stay directly browsable. Expired binary registration descriptors are not included.' if args.remaining_raw else 'Diagnostic kernel-interval JSON only. Preserve the original profiler-failure labels. Archiving does not qualify these traces as clean serving measurements.')))
    preserve()
    print('DIAGNOSTIC_ANALYSIS_ARCHIVE_VERIFIED',len(rows),archive.stat().st_size,len(parts),flush=True)


if __name__ == '__main__':
    main()
