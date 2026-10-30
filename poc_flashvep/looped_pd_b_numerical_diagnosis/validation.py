"""Read-only source and unrelated-index preservation checks."""
import hashlib
import json
import subprocess
from pathlib import Path
from .common import ROOT,REPO,write,digest


OWNED=(
    'poc_flashvep/looped_pd_b_numerical_diagnosis/',
    'poc_flashvep/reports/looped_pd_b_numerical_diagnosis/',
    'poc_flashvep/reports/looped_pd_b_loopserving_comparison.md',
    'poc_flashvep/reports/looped_pd_b_numerical_correctness.md',
    'poc_flashvep/reports/looped_pd_b_performance_root_cause.md',
    'poc_flashvep/reports/looped_pd_b_numerical_reproduce.md',
    'poc_flashvep/reports/looped_pd_b_diagnosis_final_summary.md')


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--no-write',action='store_true');args=parser.parse_args()
    locked=json.loads((ROOT/'configs/read_only_source_lock.json').read_text())
    changed=[path for path,sha in locked.items() if not Path(path).exists() or digest(path)!=sha]
    entries=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO).split(b'\0')
    outside=[r for r in entries if r and not r.split(b'\t',1)[1].decode().startswith(OWNED)]
    current=hashlib.sha256(b'\0'.join(outside)+b'\0').hexdigest()
    expected=json.loads((ROOT/'research_state.json').read_text())['original_index_sha256']
    result=dict(read_only_source_files=len(locked),changed_read_only_sources=changed,
        unrelated_index_sha256=current,expected_index_sha256=expected,
        unrelated_index_preserved=current==expected,
        HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip())
    if not args.no_write:write(ROOT/'configs/preservation_check.json',result)
    print(result)
    if changed or current!=expected:raise SystemExit(1)


if __name__=='__main__':main()
