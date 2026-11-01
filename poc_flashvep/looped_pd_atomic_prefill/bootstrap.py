"""Protect unrelated staged entries, prior source files, and the private runtime."""
import hashlib
import subprocess
from .common import *


def index_entries():
    rows=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=REPO).split(b'\0')
    return [r for r in rows if r and not any(r.split(b'\t',1)[1].decode().startswith(p) if p.endswith('/') else
        r.split(b'\t',1)[1].decode()==p for p in OWNED)]


def preserve():
    value=json.loads((ROOT/'configs/preservation_contract.json').read_text())
    rows=index_entries()
    assert hashlib.sha256(b'\0'.join(rows)+b'\0').hexdigest()==value['unrelated_index_SHA256']
    assert all(digest(REPO/p)==sha for p,sha in value['protected_files'].items())
    assert hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest()==value['RLT_diff_SHA256']
    print('PRESERVATION_PASS',len(rows),flush=True)


def main():
    setup();path=ROOT/'configs/preservation_contract.json'
    if path.exists():return preserve()
    protected={str(p.relative_to(REPO)):digest(p) for name in ('looped_pd_bottleneck_crossover','looped_pd_abc',
        'looped_pd_b_numerical_diagnosis','adaptive_recurrence_cut','adaptive_loop_token_pd_2gpu')
        for p in (REPO/'poc_flashvep'/name).rglob('*.py')}
    for name in ('looped_pd_prefill_decode_bottleneck_final_summary.md','looped_pd_phase_a_independent_capacity.md',
        'looped_pd_phase_b_bottleneck_crossover.md','looped_pd_phase_c_dynamic_a_headroom.md'):
        p=REPO/'poc_flashvep/reports'/name;protected[str(p.relative_to(REPO))]=digest(p)
    entries=index_entries()
    write(path,dict(time=time.time(),base=BASE,branch=BRANCH,start_epoch=START_EPOCH,
        unrelated_index_entries=len(entries),unrelated_index_SHA256=hashlib.sha256(b'\0'.join(entries)+b'\0').hexdigest(),
        protected_files=protected,RLT_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RLT,text=True).strip(),
        RLT_diff_SHA256=hashlib.sha256(subprocess.check_output(['git','diff','--binary'],cwd=RLT)).hexdigest()))
    write(ROOT/'state/initial_hardware.json',dict(time=time.time(),GPUs=subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version','--format=csv'],text=True),
        topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),CPU_affinity=sorted(os.sched_getaffinity(0))))
    preserve()


if __name__=='__main__':main()
