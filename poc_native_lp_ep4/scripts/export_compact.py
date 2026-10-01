"""Preserve replayable compact raw evidence; do not publish model tensors."""
import argparse
import gzip
import hashlib
from pathlib import Path
from poc_native_lp_ep4.contract import atomic_json,read_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    files=[]
    for run in args.runs:
        destination=args.output/run.name;destination.mkdir()
        for source in sorted(run.iterdir()):
            if source.suffix not in ('.json','.jsonl'):continue
            payload=source.read_bytes()
            target=destination/(source.name+'.gz')
            with target.open('wb') as f:
                f.write(gzip.compress(payload,compresslevel=9,mtime=0))
            files.append({'source':str(source),'path':str(target.relative_to(args.output)),
                'raw_sha256':hashlib.sha256(payload).hexdigest(),
                'compressed_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'raw_bytes':len(payload),'compressed_bytes':target.stat().st_size})
        files.append({'run':str(run),'status':read_json(run/'status.json'),
            'hidden_logits_tensors':'LOCAL_ONLY; numerical differences, identities and reproduction scripts are published'})
    atomic_json(args.output/'integrity.json',{'files':files,'schema':'native-lp-compact-raw-v1'})


if __name__=='__main__':main()
