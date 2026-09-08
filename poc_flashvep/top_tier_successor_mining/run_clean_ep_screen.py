"""Sequential clean baseline comparisons; no runtime policy/scheduler invention."""
import argparse,json,os,random,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from gpu_scope import allowed_devices

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
p.add_argument('--requests',type=int,default=128);p.add_argument('--reps',type=int,default=3)
p.add_argument('--run',type=int,default=0);p.add_argument('--only',help='Comma separated dataset:b values')
p.add_argument('--fixed-output',action='store_true',help='Length-controlled operator-serving diagnostic, not task-quality Pareto')
p.add_argument('--policies',type=Path,help='Explicit reference policy catalog for a corrected-port rerun')
p.add_argument('--tag',default='',help='Nonempty suffix keeps corrected-port evidence separate')
a=p.parse_args();allowed_devices();r=a.result.resolve();task=Path(__file__).resolve().parent
py='/home/esjung/.venvs/flashvep-deepep-v020/bin/python'
cpu='/home/esjung/.venvs/successor-research-audit/bin/python'
model='/home/esjung/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c'
env=dict(os.environ,OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
         SUCCESSOR_SERE_EXTENSION='/home/esjung/.cache/torch_extensions/py312_cu129/sere_successor_reference/sere_successor_reference.so')
configs=[(d,b) for d in ('chartqa','gqa') for b in (1,4,16)]
if a.only:configs=[(x.split(':')[0],int(x.split(':')[1])) for x in a.only.split(',')]
random.Random(9109+a.run).shuffle(configs)
for index,(dataset,batch) in enumerate(configs):
    name=f'ep_clean_renderer_{dataset}_b{batch}_run{a.run}'+('_fixed32' if a.fixed_output else '')+('_'+a.tag if a.tag else '')+'_20260908'
    out=r/'online'/name;assert not out.exists(),out
    command=[py,str(task/'run_ep_serving.py'),'--model',model,'--data',str(r/'data/requests.jsonl'),
        '--policies',str(a.policies.resolve() if a.policies else r/'analysis/combined_confirmatory_policies_20260908.json'),
        '--out',str(out),'--dataset',dataset,'--requests',str(a.requests),'--batch-per-dp',str(batch),
        '--output-tokens','32','--warmups','2','--repetitions',str(a.reps),
        '--order-seed',str(9171+100*a.run+index)]
    if a.fixed_output:command.append('--fixed-output')
    started=datetime.now(timezone.utc).isoformat()
    print(json.dumps(dict(started=started,name=name,command=command)),flush=True)
    with (r/'raw'/f'{name}.log').open('w') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    with (r/'raw'/'clean_screen_runs_20260908.jsonl').open('a') as f:
        f.write(json.dumps(dict(name=name,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),exit_code=result.returncode))+'\n')
    if result.returncode:raise SystemExit(result.returncode)
    with (r/'raw'/f'{name}_analysis.log').open('w') as log:
        subprocess.run([cpu,str(task/'analyze_ep_serving.py'),'--input',str(out),
            '--output',str(r/'analysis'/name),'--official-source',str(r/'references/VLMEvalKit/vlmeval/dataset/utils/vqa_eval.py')],
            check=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(completed=name)),flush=True)
