"""Record initial provenance before any experimental mutation."""
import hashlib
import subprocess
from .common import *

def main():
    setup()
    index=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO)
    contract=dict(start_utc=START_UTC,start_epoch=START_EPOCH,
        initial_index_sha256=hashlib.sha256(index).hexdigest(),
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        rlt_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RLT,text=True).strip(),
        rlt_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest(),
        inherited_sources={str(p.relative_to(REPO)):digest(p) for folder in
            ('adaptive_recurrence_cut','looped_pd_abc','looped_pd_b_numerical_diagnosis')
            for p in (REPO/'poc_flashvep'/folder).glob('*.py')},
        spec_sha256=digest(REPO/'poc_flashvep/reports/adaptive_loop_token_pd_2gpu_w1_w4_12h_poc_spec.md'))
    path=ROOT/'configs/initial_contract.json'
    if path.exists():raise FileExistsError(path)
    write(path,contract);budget(status='AUDIT',selected_matrix='UNDECIDED')
    write(ROOT/'state/initial_gpu.json',dict(query=subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)))

if __name__=='__main__':main()
