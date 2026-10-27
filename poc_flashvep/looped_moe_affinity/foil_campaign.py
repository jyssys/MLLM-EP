"""Wait for the native IFM lane, then run available full Foil paper anchors."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
from .common import *
from .launch import launch
from . import resources

def main():
    # Once all native jobs started, their lanes have no queued work to steal back.
    while len(list((ROOT/'raw').glob('ifm_*_official_*')))<14:
        time.sleep(5)
    tasks=['lambada_standard','lambada_openai','xwinograd_en']
    configs=[dict(id=f'foil1_official_{task}_full_v1',model='foil1',phase='foil_official',tasks=[task],
        python='/home/esjung/.venvs/looped-moe-affinity-eval-py312/bin/python',timeout_s=14400) for task in tasks]
    result=[];pending=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        while configs or pending:
            for future,gpu in list(pending):
                if future.done():
                    value=future.result();result.append(value);pending.remove((future,gpu))
                    write(ROOT/'foil_quality_campaign.json',result);print(value,flush=True)
            occupied={gpu for _,gpu in pending}
            available=[gpu for gpu in resources.burners() if gpu not in occupied]
            while configs and available:
                gpu=available.pop(0);config=dict(configs.pop(0),gpus=[gpu])
                pending.append((pool.submit(launch,config),gpu))
            if configs or pending:time.sleep(2)

if __name__=='__main__':main()
