"""Resume-safe staged campaign. Quality jobs can share disjoint EP4 groups."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
from .common import ROOT,DATA,MODELS,write
from .launch import launch

EVAL_PYTHON='/home/esjung/.venvs/looped-moe-affinity-eval-py312/bin/python'

def run(config):
    write(ROOT/'research_state.json',dict(status='RUNNING',current=config,updated=time.time(),
        resume='python -u -m poc_flashvep.looped_moe_affinity.full_campaign',
        warning='A completed stage is not a completed PoC. Do not infer results from queued configurations.'))
    try:result=launch(config)
    except Exception as error:
        result=dict(id=config['id'],status='FAILED',error=repr(error))
        write(ROOT/'runs'/f'{config["id"]}.failure.json',result)
        print(result,flush=True)
        raise
    print(result,flush=True)
    return result

def quality(model,gpus):
    tasks=['arc_challenge']+['mmlu/'+p.name for p in sorted((DATA/'paper_part1/mmlu').iterdir()) if p.is_dir()]
    config=dict(id=f'{model}_ep{len(gpus)}_official_quality_v1',model=model,gpus=gpus,
        phase='ep_eval',tasks=tasks,timeout_s=86400)
    if model=='foil1':config.update(tasks=['lambada_standard','lambada_openai','xwinograd_en'],python=EVAL_PYTHON)
    return config

def main():
    # An earlier smoke dispatcher owns the devices until its durable result exists.
    while not (ROOT/'ep_smoke_campaign.json').exists():time.sleep(10)
    smoke=json.loads((ROOT/'ep_smoke_campaign.json').read_text())
    assert len(smoke)==6 and all(v['status'] in ['PASS','ALREADY_COMPLETE'] for v in smoke),smoke
    # Native routes are evidence collection, not clean latency measurements.
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(run,[dict(id=f'{m}_route_corpus_v2',model=m,gpus=[i],phase='routes',timeout_s=14400)
            for i,m in enumerate(MODELS)]))
    from .aggregate_quality import collect
    collect()
    # Headline timing: one experiment group at a time. Owned burners use unused devices.
    for ep in [4,8]:
        for model in MODELS:
            run(dict(id=f'{model}_ep{ep}_full_profile_v1',model=model,gpus=list(range(ep)),phase='profile',
                batches=[1,2,4,8,16],lengths=[128,512,2048],warmups=10,repeats=5,
                generation_cap=32,generation_repeats=3,timeout_s=43200))
    # Full official evaluations. These jobs do not produce clean serving latency claims.
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(run,quality('ifm_ouro',list(range(4)))),
            pool.submit(run,quality('ifm_huginn',list(range(4,8))))]
        for f in futures:f.result()
    run(quality('foil1',list(range(4))))
    for model in MODELS:run(quality(model,list(range(8))))
    # Rebuild uses the original official held-out recipe. Do not score an incomplete corpus.
    data=DATA/'foil-heldout/fineweb_edu_heldout_v1'
    deadline=time.monotonic()+3600
    while not (data/'manifest.json').exists():
        assert time.monotonic()<deadline,'Official held-out builder did not finish. Preserve its log.'
        time.sleep(15)
    for ep in [1,4,8]:
        run(dict(id=f'foil1_ep{ep}_official_fineweb_nll_v1',model='foil1',gpus=list(range(ep)),phase='nll',
            nll_data=str(data),python=EVAL_PYTHON,timeout_s=21600))
    collect()
    write(ROOT/'research_state.json',dict(status='GPU_CAMPAIGN_COMPLETE_ANALYSIS_REQUIRED',updated=time.time(),
        next='Aggregate traces, validate paired quality, write the complete reports, test, commit, and push.'))

if __name__=='__main__':main()
