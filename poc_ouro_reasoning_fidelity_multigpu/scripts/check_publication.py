"""Check owned Git candidates without printing environment secret values."""
import io
import os
import re
import subprocess
import tarfile
import zipfile
import argparse
from poc_ouro_reasoning_fidelity_multigpu.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/publication_check.json')
    options=parser.parse_args()
    command=['git','ls-files','--others','--exclude-standard','--',ROOT.name]
    candidates=subprocess.check_output(command,cwd=PROJECT,text=True).splitlines()
    staged=subprocess.check_output(['git','diff','--cached','--name-only','--',ROOT.name],cwd=PROJECT,text=True).splitlines()
    paths=[PROJECT/name for name in sorted(set(candidates+staged))]
    assert paths
    assert all(path.is_relative_to(ROOT) for path in paths)
    secrets=[value.encode() for key,value in os.environ.items()
             if re.search(r'TOKEN|SECRET|PASSWORD|API_KEY',key,re.I) and len(value)>=8]
    matches=[]
    large=[]
    def scan(label,data):
        if any(value in data for value in secrets):
            matches.append(label)
    for path in paths:
        label=str(path.relative_to(PROJECT))
        if path.stat().st_size>=100_000_000:
            large.append(label)
        data=path.read_bytes()
        scan(label,data)
        if path.suffix=='.zip':
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.namelist():
                    scan(label+'::'+member,archive.read(member))
        elif path.name.endswith('.tar.gz'):
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
                for member in archive:
                    if member.isfile():
                        scan(label+'::'+member.name,archive.extractfile(member).read())
    raw_paths=[ROOT/'results/environment.json',*sorted((ROOT/'results').rglob('clean_start.json')),
               *sorted((ROOT/'results').rglob('*.npz')),
               *sorted((ROOT/'results/quality_v1').glob('*_wave*.json'))]
    assert not set(paths)&set(raw_paths)
    target=(ROOT/options.output).resolve()
    assert target.is_relative_to(ROOT/'results')
    assert not target.exists()
    write(target,dict(owned_candidate_files=len(paths),candidate_bytes=sum(path.stat().st_size for path in paths),
        environment_secret_value_matches=matches,files_at_or_above_100MB=large,
        raw_private_or_off_Git_files_excluded=len(raw_paths),
        secret_values_printed_or_stored=False,
        scope='Checks candidate bytes and compressed members against current secret-valued environment variables. Not a universal secret-detection guarantee.'))
    assert not matches and not large
    print('Publication check passes. No current environment-secret value occurs in the checked Git candidates.')


if __name__=='__main__':
    main()
