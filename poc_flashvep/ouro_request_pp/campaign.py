"""Persist commands and logs for bounded physical experiments."""
import argparse
import subprocess
import time
from .contract import *

PYTHON = '/home/esjung/.venvs/llada2-mini-reference/bin/python'


def execute(name, args):
    target = ROOT/'logs'/f'{name}.log'
    target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(), target
    command = [PYTHON,'-u','-m','poc_flashvep.ouro_request_pp.benchmark']+args
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='4,5,6,7',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    start = time.time()
    write(ROOT/'commands'/f'{name}.json',dict(argv=command,environment={k:env[k] for k in
        ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE']},
        cwd=str(PROJECT),start_epoch_s=start,
        source_sha256={str(p.relative_to(PROJECT)):sha(p) for p in (PROJECT/'poc_flashvep/ouro_request_pp').glob('*.py')}))
    with target.open('w') as log:
        p = subprocess.Popen(command,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
        code = p.wait()
    write(ROOT/'commands'/f'{name}_completion.json',dict(returncode=code,wall_s=time.time()-start))
    print('DONE',name,code,round(time.time()-start,1),flush=True)
    assert code==0, f'{name}: inspect {target}'


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['correctness','correctness_graph','extended','probe','primary','trace'])
    a=p.parse_args()
    if a.phase=='correctness':
        execute('correctness_v1',['--mode','correctness'])
    elif a.phase=='correctness_graph':
        execute('correctness_graph_v5',['--mode','correctness','--graphs'])
    elif a.phase=='extended':
        for P,C in [(128,4),(512,2)]:
            execute(f'correctness_P{P}_C{C}_v5',['--mode','correctness','--graphs',
                '--correctness-prompt',str(P),'--correctness-count',str(C),'--correctness-depths','4'])
    elif a.phase=='probe':
        for arm in ['layer_pp4','loop_pp4']:
            execute(f'probe_{arm}_v5',['--mode','benchmark','--placement',arm,'--label',f'probe_{arm}_v5',
                '--depth','4','--concurrency','4','--prompt-len','128','--output-len','8','--trace','--graphs'])
    elif a.phase=='trace':
        for arm in ['layer_pp4','loop_pp4']:
            execute(f'trace_{arm}_v5',['--mode','benchmark','--placement',arm,'--label',f'trace_{arm}_v5',
                '--depth','4','--concurrency','4','--prompt-len','512','--output-len','8','--trace','--graphs'])
    else:
        assert (ROOT/'correctness_checks_graph_v5.json').exists()
        assert all(r['PASS'] for r in json.loads((ROOT/'correctness_checks_graph_v5.json').read_text()))
        for repeat in range(3):
            cases = []
            for P in [128,512]:
                for O in [32,128]:
                    for C in [1,4,8,16]:
                        cases.append(dict(depth=4,C=C,P=P,O=O,repeat=repeat))
            for depth in [2,3]:
                for C in [2,8,16]:
                    cases.append(dict(depth=depth,C=C,P=128,O=32,repeat=repeat))
            for C in [2,32]:
                cases.append(dict(depth=4,C=C,P=128,O=32,repeat=repeat))
            for C in [4,8]:
                cases.append(dict(depth=4,C=C,P=2048,O=32,repeat=repeat))
            cases.append(dict(depth=4,C=8,P=-1,O=32,repeat=repeat))
            plan = ROOT/f'plan_v5_r{repeat}.json'
            write(plan,dict(cases=cases))
            for arm in (['layer_pp4','loop_pp4'] if repeat%2==0 else ['loop_pp4','layer_pp4']):
                name=f'primary_{arm}_v5_r{repeat}'
                execute(name,['--mode','benchmark','--placement',arm,'--label',name,'--case-file',str(plan),'--graphs'])


if __name__=='__main__':main()
