"""Preserve every unrelated Git index entry and inherited source file."""
import hashlib
import json
import subprocess
from .common import *

OWNED=['poc_flashvep/adaptive_loop_token_pd_2gpu/','poc_flashvep/reports/adaptive_loop_token_pd_2gpu/']
OWNED += ['poc_flashvep/reports/'+name for name in (
    'adaptive_loop_token_pd_2gpu_w1_w4_12h_poc_spec.md',
    'adaptive_loop_token_pd_2gpu_correctness_and_protocol.md',
    'adaptive_loop_token_pd_2gpu_w1_w4_matrix.md',
    'adaptive_loop_token_pd_2gpu_online_policy_and_slo.md',
    'adaptive_loop_token_pd_2gpu_final_summary.md')]

def main():
    original=json.loads((ROOT/'configs/initial_contract.json').read_text())
    entries=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO).split(b'\0')
    kept=[]
    for entry in entries:
        if not entry:continue
        path=entry.split(b'\t',1)[1].decode()
        if not any(path.startswith(x) if x.endswith('/') else path==x for x in OWNED):kept.append(entry)
    current=hashlib.sha256(b'\0'.join(kept)+b'\0').hexdigest()
    if current!=original['initial_index_sha256']:raise RuntimeError('Unrelated staged entries changed')
    for path,expected in original['inherited_sources'].items():
        if digest(REPO/path)!=expected:raise RuntimeError('Inherited source changed: '+path)
    actual=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest()
    if actual!=original['rlt_diff_sha256']:raise RuntimeError('Private RLT changes are not preserved')
    write(ROOT/'state/preservation_check.json',dict(passed=True,time=time.time(),index_entries=len(kept),
        index_sha256=current,inherited_files=len(original['inherited_sources']),rlt_diff_sha256=actual))
    print('PRESERVED_UNRELATED_INDEX_AND_INHERITED_SOURCES',len(kept))

if __name__=='__main__':main()
