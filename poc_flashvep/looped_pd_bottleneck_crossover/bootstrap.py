"""Preserve the user's index and previous evidence before experiments."""
import hashlib
import subprocess
import zipfile
from .common import *

def index_entries():
    rows=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO).split(b'\0')
    kept=[]
    for row in rows:
        if not row:continue
        name=row.split(b'\t',1)[1].decode()
        if not any(name.startswith(x) if x.endswith('/') else name==x for x in OWNED):kept.append(row)
    return kept

def preserve():
    value=json.loads((ROOT/'configs/initial_contract.json').read_text());entries=index_entries()
    assert hashlib.sha256(b'\0'.join(entries)+b'\0').hexdigest()==value['unrelated_index_sha256']
    assert all(digest(REPO/p)==sha for p,sha in value['protected_files'].items())
    assert digest(SPEC)==value['spec_sha256']
    assert hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest()==value['rlt_diff_sha256']
    print('PRESERVATION_PASS',len(entries))

def main():
    setup();path=ROOT/'configs/initial_contract.json'
    if path.exists():return preserve()
    protected={str(p.relative_to(REPO)):digest(p) for folder in ('adaptive_loop_token_pd_2gpu',
        'adaptive_recurrence_cut','looped_pd_abc','looped_pd_b_numerical_diagnosis')
        for p in (REPO/'poc_flashvep'/folder).glob('*.py')}
    for name in ('adaptive_recurrence_cut_final_summary.md','adaptive_loop_token_pd_2gpu_final_summary.md','adaptive_loop_token_pd_2gpu_w1_w4_matrix.md'):
        p=REPO/'poc_flashvep/reports'/name;protected[str(p.relative_to(REPO))]=digest(p)
    entries=index_entries()
    write(path,dict(start_utc=START_UTC,start_epoch=START_EPOCH,branch=BRANCH,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        unrelated_index_sha256=hashlib.sha256(b'\0'.join(entries)+b'\0').hexdigest(),
        unrelated_index_entries=len(entries),protected_files=protected,spec_sha256=digest(SPEC),
        model_pins=MODEL_PINS,Qwen_pin=QWEN_PIN,
        rlt_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RLT,text=True).strip(),
        rlt_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest()))
    with zipfile.ZipFile(ROOT/'configs/frozen_input_spec.zip','w',compression=zipfile.ZIP_DEFLATED) as z:z.write(SPEC,SPEC.name)
    write(ROOT/'state/initial_gpu.json',dict(time=time.time(),GPU_query=subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        CPU_affinity=sorted(os.sched_getaffinity(0))))
    budget(status='AUDIT')
    preserve()

if __name__=='__main__':main()
