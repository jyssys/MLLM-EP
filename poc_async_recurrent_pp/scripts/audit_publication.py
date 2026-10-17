"""Check candidate artifacts for environment credential values and large files."""
import argparse
import os
import re
import subprocess
import zipfile
from poc_async_recurrent_pp.contract import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    values=[value.encode() for key,value in os.environ.items()
            if re.search(r'TOKEN|SECRET|PASSWORD|API.?KEY|CREDENTIAL',key,re.I) and len(value)>=12]
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',ROOT.name],cwd=PROJECT,text=True)
    violations=[];oversized=[];count=0;total=0;members=0
    for name in sorted(set(names.splitlines())):
        path=PROJECT/name
        if not path.is_file():
            continue
        raw=path.read_bytes();count+=1;total+=len(raw)
        if len(raw)>=100_000_000:
            oversized.append(name)
        if any(value in raw for value in values):
            violations.append(name)
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    members+=1
                    if any(value in archive.read(member) for value in values):
                        violations.append(name+'::'+member)
    write(args.output,dict(candidate_files=count,total_bytes=total,archive_members=members,
        credential_values_checked=len(values),credential_match_paths=violations,
        oversized_files=oversized,pass_check=not violations and not oversized,
        scope='Current environment credential values only. No secret value is logged. Ignored private GPU inventories remain local.'))
    print(dict(files=count,bytes=total,archive_members=members,credential_matches=len(violations),oversized=len(oversized)))
    assert not violations and not oversized


if __name__=='__main__':
    main()
