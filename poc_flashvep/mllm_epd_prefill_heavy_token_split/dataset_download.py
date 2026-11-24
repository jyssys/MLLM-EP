"""Download pinned image-only source snapshots before selecting requests."""
import json
from pathlib import Path
import time
import re
from .common import ROOT,write,digest,initialize,preserve

PATTERNS={
    'TIGER-Lab/Mantis-Eval':['mantis_eval/test-*.parquet'],
    'MUIRBENCH/MUIRBENCH':['data/test-*.parquet'],
    'lmms-lab/MP-DocVQA':['data/val-*.parquet'],
    'ahmed-masry/ChartQA':['data/test-*.parquet'],
    'lmms-lab/DocVQA':['DocVQA/validation-*.parquet','InfographicVQA/validation-*.parquet'],
    'yubo2333/MMLongBench-Doc':['data/train-*.parquet','documents/*.pdf'],
}


def main():
    initialize()
    from huggingface_hub import snapshot_download,hf_hub_download,HfApi
    from concurrent.futures import ThreadPoolExecutor
    results=[]
    for repo,patterns in PATTERNS.items():
        name=repo.replace('/','__');receipt=ROOT/'state'/('download_'+name+'.json')
        if receipt.exists():results.append(json.loads(receipt.read_text()));continue
        meta=json.loads((ROOT/'references/datasets'/(name+'.json')).read_text());assert meta['accessible']
        started=time.time();print('DATASET_DOWNLOAD_START',repo,meta['revision'],flush=True)
        failures=[]
        if repo=='yubo2333/MMLongBench-Doc':
            # An upstream PDF currently has a different response size from its pinned LFS pointer.
            # Preserve that failure. Do not accept the replacement as the pinned source.
            info=HfApi().dataset_info(repo,revision=meta['revision'],files_metadata=True)
            selected=[s for s in info.siblings if s.rfilename.endswith('.pdf') or s.rfilename.endswith('.parquet') or s.rfilename=='README.md']
            def fetch(s):
                try:
                    p=Path(hf_hub_download(repo,s.rfilename,repo_type='dataset',revision=meta['revision']))
                    if s.lfs:assert digest(p)==s.lfs.sha256,'Pinned LFS SHA256 mismatch'
                    return p,None
                except Exception as error:
                    return None,dict(filename=s.rfilename,error=type(error).__name__+': '+str(error))
            with ThreadPoolExecutor(max_workers=4) as pool:
                fetched=list(pool.map(fetch,selected))
            failures=[error for _,error in fetched if error]
            directory=next(p for p,error in fetched if p and p.suffix=='.parquet').parents[1]
            write(ROOT/'state'/('download_source_failures_'+name+'.json'),dict(time=time.time(),failures=failures))
        for attempt in range(0 if repo=='yubo2333/MMLongBench-Doc' else 4):
            try:
                directory=Path(snapshot_download(repo,repo_type='dataset',revision=meta['revision'],
                    allow_patterns=patterns+['README.md','LICENSE','LICENSE*'],max_workers=4))
                break
            except OSError as error:
                match=re.search(r'\(([^()]+\.pdf)\)',str(error))
                append_failure=dict(time=time.time(),repo=repo,attempt=attempt,error=str(error))
                write(ROOT/'state'/f'download_retry_{name}_{attempt}.json',append_failure)
                if not match or attempt==3:raise
                filename='documents/'+Path(match.group(1)).name
                print('FORCE_RETRY_SINGLE_FILE',repo,filename,flush=True)
                hf_hub_download(repo,filename,repo_type='dataset',revision=meta['revision'],force_download=True)
        files=[dict(name=str(p.relative_to(directory)),bytes=p.stat().st_size,SHA256=digest(p))
            for p in sorted(directory.rglob('*')) if p.is_file()]
        row=dict(time=time.time(),repo=repo,revision=meta['revision'],directory=str(directory),
            elapsed_seconds=time.time()-started,files=files,passed=not failures,source_failures=failures,
            scope='Available files verified against pinned revision. Unverifiable files are excluded and retained in the failure ledger.')
        write(receipt,row);results.append(row);print('DATASET_DOWNLOAD_COMPLETE',repo,len(files),flush=True)
    write(ROOT/'state/dataset_download_complete.json',dict(time=time.time(),passed=True,sources=results,
        scope='Official source or explicitly named mirror snapshots. No performance-based sample filtering. '
            'Parquet media and reference text remain local. No video source was downloaded.'))
    preserve()


if __name__=='__main__':main()
