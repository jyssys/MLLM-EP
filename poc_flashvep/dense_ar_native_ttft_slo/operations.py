"""Reuse audited closeout,lossless archives and private-index publication."""
import argparse
import gzip
import json
import re
import shutil
import subprocess
import time
from .common import ROOT,SOURCE,REPO,BASE,BRANCH,REPORT_NAMES,write,digest,preserve
from .runtime import bind


def close():
    bind()
    from ..dense_ar_p_critical_path.resource_closeout import close as audited_close
    return audited_close()


def archive():
    bind()
    from ..dense_ar_p_critical_path.evidence import archive as audited_archive
    result=audited_archive()
    compress_csvs()
    return result


def compress_csvs():
    """Retain original tables and verify every existing gzip before reuse."""
    preserve()
    compressed=[]
    for path in sorted((ROOT/'figures').glob('*.csv')):
        if path.stat().st_size<80*1024**2:continue
        target=path.with_suffix('.csv.gz')
        if not target.exists():
            with path.open('rb') as source,target.open('wb') as destination:
                with gzip.GzipFile(fileobj=destination,mode='wb',filename='',mtime=0) as output:
                    shutil.copyfileobj(source,output)
        import hashlib
        checksum=hashlib.sha256()
        with gzip.open(target,'rb') as stream:
            for chunk in iter(lambda:stream.read(1024**2),b''):checksum.update(chunk)
        assert checksum.hexdigest()==digest(path)
        compressed.append(dict(original=str(path.relative_to(REPO)),original_SHA256=digest(path),
            gzip=str(target.relative_to(REPO)),gzip_SHA256=digest(target),verified_lossless=True))
    write(ROOT/'archives/large_derived_csv_gzip_manifest.json',dict(time=time.time(),passed=True,files=compressed,
        scope='Deterministic lossless copies of large derived CSV files. Original CSV and raw evidence remain unchanged.'))
    return compressed


def selected_paths():
    bind();preserve()
    from ..dense_ar_p_critical_path import publication
    publication.SOURCE=SOURCE;publication.NAMES=REPORT_NAMES
    paths=publication.selected_paths()
    paths=[path for path in paths if (ROOT in path.parents or SOURCE in path.parents or
        (path.parent==ROOT.parent and path.name in REPORT_NAMES)) and
        not (path.suffix=='.csv' and path.with_suffix('.csv.gz').exists())]
    paths.extend((ROOT/'figures').glob('*.csv.gz'))
    paths.extend((ROOT/'archives').glob('*_gzip_manifest.json'))
    paths.append(ROOT/'research_decision_log.md')
    paths.append(publication_links(paths))
    assert all(path.exists() and path.stat().st_size<90*1024**2 for path in paths)
    record=ROOT/'state/publication_selection.json'
    value=json.loads(record.read_text());value.update(selected=len(set(paths)),
        scope='New Native-relative TTFT/Joint-SLO namespace only. Preserve the user checkout,index and old reports. '
            'Publish verified public raw parts,canonical sources,five reports,figures and their tables. '
            'Large derived CSVs use verified gzip copies. Original contexts,weights and private full logs remain local.')
    write(record,value);return sorted(set(paths))


def publication_links(paths):
    """Check report links against selected blobs or the immutable parent tree."""
    chosen={path.resolve() for path in paths};rows=[]
    for report in [ROOT.parent/name for name in REPORT_NAMES]+[ROOT/'README.md']:
        if not report.exists():continue
        for link in re.findall(r'\]\(([^)]+)\)',report.read_text()):
            if '://' in link or link.startswith('#'):continue
            target=(report.parent/link.split('#')[0]).resolve()
            relative=str(target.relative_to(REPO))
            inherited=False
            if target not in chosen:
                inherited=subprocess.run(['git','cat-file','-e',BASE+':'+relative],cwd=REPO,
                    stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
            rows.append(dict(report=str(report.relative_to(REPO)),target=relative,
                selected=target in chosen,inherited_from_base=inherited,passed=target in chosen or inherited))
    value=dict(passed=all(row['passed'] for row in rows),links=rows,
        scope='A locally existing artifact is not sufficient. Each report link must be selected for publication or inherited from the immutable base tree.')
    path=ROOT/'state/publication_link_audit.json';write(path,value)
    assert value['passed'],[row for row in rows if not row['passed']]
    return path


def publish():
    bind();preserve()
    from ..dense_ar_p_critical_path.publication import commit_paths,push_record
    record='publication_commit_science.json';path=ROOT/'state'/record
    if path.exists():
        value=json.loads(path.read_text())
        if not value['passed']:return push_record(record)
        raise AssertionError('Science commit already verified. Use an explicit closeout follow-up,not an accidental duplicate publication.')
    return commit_paths(selected_paths(),
        'Dense AR Native-relative TTFT and Joint SLO: request-level failure attribution and controlled policy evidence',record)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('close','archive','compress_csvs','select','publish'));args=parser.parse_args()
    value=selected_paths() if args.action=='select' else globals()[args.action]()
    print(len(value) if isinstance(value,list) else value.get('commit',value.get('passed')) if isinstance(value,dict) else value,flush=True)
