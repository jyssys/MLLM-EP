"""Resume-safe native anchor evaluation. Model workers use distinct physical GPUs."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import queue
from .common import *
from .launch import launch

def native_plan():
    path=DATA/'paper_part1/mmlu'
    sizes={p.parent.name:p.read_bytes().count(b'\n') for p in path.glob('*/test.jsonl')}
    assert len(sizes)==57 and sum(sizes.values())==14042
    buckets=[[] for _ in range(6)];loads=[0]*6
    for subject,n in sorted(sizes.items(),key=lambda item:-item[1]):
        i=min(range(6),key=lambda i:loads[i]);buckets[i].append('mmlu/'+subject);loads[i]+=n
    configs=[]
    for model in ['ifm_ouro','ifm_huginn']:
        configs.append(dict(id=f'{model}_official_arc_full_v1',model=model,phase='official_native',
            tasks=['arc_challenge'],items=1165,timeout_s=14400))
        for i,tasks in enumerate(buckets):
            configs.append(dict(id=f'{model}_official_mmlu_bucket{i}_v1',model=model,phase='official_native',
                tasks=tasks,items=loads[i],timeout_s=21600))
    write(ROOT/'runs/native_quality_plan.json',dict(configs=configs,mmlu_partition='Whole official subjects. No within-subject or prompt changes.',
        unavailable='HellaSwag official distributions unavailable. See dataset_manifest.json.'))
    return sorted(configs,key=lambda c:-c['items'])

def main():
    setup();jobs=queue.Queue()
    for config in native_plan():jobs.put(config)
    results=[]
    import threading
    lock=threading.Lock()
    def lane(gpu):
        while True:
            try:config=dict(jobs.get_nowait(),gpus=[gpu])
            except queue.Empty:return
            try:result=launch(config)
            except Exception as error:result=dict(id=config['id'],status='FAILED',error=repr(error))
            with lock:
                results.append(result);write(ROOT/'native_quality_campaign.json',results)
                print(result,flush=True)
            jobs.task_done()
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lane,range(8)))

if __name__=='__main__':main()
