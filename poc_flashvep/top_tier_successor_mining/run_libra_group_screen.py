"""Sequential author-runtime VL groups; captured-vision prefill, not full E2E."""
import argparse,json,os,random,subprocess
from pathlib import Path
from gpu_scope import allowed_devices

p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True)
p.add_argument('--groups',default='gqa_1,gqa_2,gqa_3,chartqa_0,chartqa_1,chartqa_2,chartqa_3')
p.add_argument('--reps',type=int,default=10);a=p.parse_args();allowed_devices()
r=a.result.resolve();task=Path(__file__).resolve().parent
py='/home/esjung/.venvs/libra-supplement-py310/bin/python'
cpu='/home/esjung/.venvs/successor-research-audit/bin/python'
env=dict(os.environ,CUDA_HOME='/usr/local/cuda-12.8',HF_HUB_OFFLINE='1',
    PATH='/home/esjung/.venvs/libra-supplement-py310/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin')
groups=a.groups.split(',');random.Random(7317).shuffle(groups)
for group in groups:
    name=f'libra_native_vl48_{group}_frozen_control_20260908';out=r/'online'/name
    assert not out.exists(),out
    command=[py,str(task/'libra_native_sanity.py'),'--model',str(r/'references/qwen_vl_text_descriptor'),
        '--data',str(r/'data/fineweb.jsonl'),'--out',str(out),'--vl-inputs',str(r/'quality/libra_vl_bridge_inputs'),
        '--vl-group',group,'--cases','1','--warmup','3','--reps',str(a.reps),'--frozen-route-oracle']
    print(json.dumps(dict(name=name,command=command)),flush=True)
    with (r/'raw'/f'{name}.log').open('w') as log:
        subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    with (r/'raw'/f'{name}_analysis.log').open('w') as log:
        subprocess.run([cpu,str(task/'analyze_libra_native.py'),'--input',str(out),
            '--output',str(r/'analysis'/name)],check=True,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps(dict(completed=name)),flush=True)
