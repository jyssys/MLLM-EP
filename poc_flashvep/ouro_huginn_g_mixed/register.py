"""Freeze the contract before pretrained measurements."""
import subprocess
from .common import ROOT, REPO, SPEC, FORK, MODELS, DEPTHS, UPSTREAM, write, digest

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / 'registration.json').exists()
    write(ROOT / 'registration.json', dict(spec=str(SPEC), spec_sha256=digest(SPEC),
        parent=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source=UPSTREAM, fork=str(FORK), models={k: str(v) for k,v in MODELS.items()},
        fixed_depth=DEPTHS, primary_chunk=512, secondary_chunks=[256,1024],
        numerical_tolerance=dict(relative_L2=0.03, absolute_max=0.125),
        support_gate='Both pretrained models must qualify legal split/group/Mixed before method promotion.',
        progression='P2 opposing complete-token crossover, >=3%, repeated process starts. No forced adaptive implementation.',
        capacity='1000 offered per policy/rate, >=3 starts, >=90% attainment and stable queue.',
        adaptive='NOT_IMPLEMENTED before P2', quality='NOT_RUN before support qualification'))
    commands = dict(gpus=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,clocks.sm','--format=csv'],
        topology=['nvidia-smi','topo','-m'], processes=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv'])
    for name, command in commands.items():
        write(ROOT / 'environment' / f'{name}.json', dict(command=command,
            output=subprocess.check_output(command, text=True)))
    jobs = [dict(id=f'support_{name}_s0', kind='support', model=name, gpu=gpu,
        P=128, C=64, cap=8, timeout_s=3600) for name,gpu in [('ouro',0),('huginn',1)]]
    write(ROOT/'plans'/'support.json',dict(gpus=list(range(8)),jobs=jobs))

if __name__ == '__main__': main()
