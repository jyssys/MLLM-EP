"""Preserve lossless profiler exports without changing the source SQLite files."""
import gzip
import shutil
import sqlite3
from .contract import *


def main():
    stage=read(ROOT/'state/continuation_stage.json')
    assert stage.get('status') in ['NEEDS_REVIEW','FINAL_REVIEW_COMPLETE'],'Wait for the GPU and SQLite-export stages to finish.'
    records=[]
    for source in sorted((ROOT/'profiles').glob('*.sqlite')):
        with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as database:
            assert database.execute('PRAGMA integrity_check').fetchone()[0]=='ok',source
        destination=source.with_suffix('.sqlite.gz')
        if not destination.exists():
            with source.open('rb') as input_,destination.open('wb') as output:
                with gzip.GzipFile(filename=source.name,mode='wb',fileobj=output,mtime=0) as compressed:
                    shutil.copyfileobj(input_,compressed,length=2**20)
        digest=hashlib.sha256()
        with gzip.open(destination,'rb') as restored:
            for chunk in iter(lambda:restored.read(2**20),b''):digest.update(chunk)
        assert digest.hexdigest()==sha(source),'Lossless archive verification failed.'
        assert destination.stat().st_size<90*2**20,'Review the Git file-size limit before staging this archive.'
        rep=source.with_suffix('.nsys-rep')
        records.append(dict(label=source.stem,SQLite_source=str(source.relative_to(ROOT)),
            SQLite_bytes=source.stat().st_size,SQLite_sha256=sha(source),
            lossless_export=str(destination.relative_to(ROOT)),compressed_bytes=destination.stat().st_size,
            compressed_sha256=sha(destination),decompressed_sha256=digest.hexdigest(),integrity_check='ok',
            native_Nsight_report=str(rep.relative_to(ROOT)) if rep.exists() else None,
            native_report_sha256=sha(rep) if rep.exists() else None,
            raw_scope='Full SQLite export, not only the filtered timeline. Native Nsight containers remain local and reproducible.'))
    write(ROOT/'profile_archive_manifest.json',dict(exports=records,
        instructions='Decompress each .sqlite.gz file to restore its full raw SQLite export. Verify the listed SHA256.',
        preservation='The original SQLite and Nsight report files remain unchanged on the research server.'))
    print('VERIFIED_LOSSLESS_PROFILE_EXPORTS',len(records),flush=True)


if __name__=='__main__':main()
