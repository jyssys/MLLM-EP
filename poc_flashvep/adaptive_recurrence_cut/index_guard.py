"""Check that unrelated staged entries remain byte-identical."""
import hashlib
import json
import subprocess
from .common import ROOT,REPO,OWNED,write


def main():
    data=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)
    original=json.loads((ROOT/'configs/source_contract.json').read_text())['initial_git_index_sha256']
    # The initial checksum used newline-delimited Git output. Preserve that format.
    kept=[]
    for entry in data.split(b'\0'):
        if not entry:continue
        path=entry.split(b'\t',1)[1].decode()
        if any(path.startswith(p) if p.endswith('/') else path==p for p in OWNED):continue
        kept.append(entry+b'\n')
    actual=hashlib.sha256(b''.join(kept)).hexdigest()
    if actual!=original:raise RuntimeError(f'Unrelated Git index changed: {actual} != {original}')
    write(ROOT/'state/unrelated_index_preservation.json',dict(passed=True,initial_sha256=original,current_filtered_sha256=actual))
    print('UNRELATED_INDEX_BYTE_IDENTICAL',len(kept))


if __name__=='__main__':main()
