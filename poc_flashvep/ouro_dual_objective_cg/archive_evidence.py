"""Archive immutable raw attempts and logs after all registered jobs finish."""
import json
import tarfile
from .common import ROOT,write,digest

def main():
    validation=json.loads((ROOT/'evidence_validation.json').read_text())
    assert validation['all_complete'],'Finish and validate all registered attempts first'
    dest=ROOT/'archives';dest.mkdir(exist_ok=True)
    archive=dest/'dual_objective_raw_attempts_v1.tar.gz'
    assert not archive.exists(),'Preserve an existing archive and register a new version'
    records=[]
    for directory in ('raw','logs'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file():records.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path)))
    write(ROOT/'raw_evidence_SHA256.json',dict(files=records,
        scope='All raw attempts, including failed v1 attempts and their logs. No model weights or environment packages.'))
    with tarfile.open(archive,'w:gz',compresslevel=6) as target:
        for row in records:target.add(ROOT/row['path'],arcname=row['path'],recursive=False)
    # Validate file hashes again to reject changes during archive creation.
    assert all(digest(ROOT/row['path'])==row['sha256'] for row in records)
    write(ROOT/'archive_manifest.json',dict(local_archive=str(archive),sha256=digest(archive),
        compressed_bytes=archive.stat().st_size,uncompressed_file_bytes=sum(r['bytes'] for r in records),
        files=len(records),file_manifest_sha256=digest(ROOT/'raw_evidence_SHA256.json'),
        source_files_retained=True,scope='Local portable raw archive. Git contains code, reports, frozen manifests, and compressed analysis summaries.'))
    print(json.dumps(json.loads((ROOT/'archive_manifest.json').read_text()),indent=2))

if __name__=='__main__':main()
