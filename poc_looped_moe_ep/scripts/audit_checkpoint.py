"""Public release/architecture audit, no GPU or speculative model construction."""
import datetime
import importlib.metadata
import subprocess
import requests
from poc_looped_moe_ep.contract import *


def get(url):
    response=requests.get(url,timeout=45)
    return response


def main():
    candidates=[]
    for repo,subfolder in [(MODEL,SUBFOLDER),
        ('ml-ryanlee/looped-moe-16x2-e8k2-32L-d512-1e18-a100',''),
        ('IFM/LoopedLM-P1-moe-middle-500b',''),('IFM/LoopedLM-P1-moe-huginn-500b',''),
        ('IQuestLab/Loopie-20B-A2B-preview',''),('IQuestLab/Loopie-6B-A0.6B-preview','')]:
        response=get('https://huggingface.co/api/models/'+repo+'?blobs=true')
        record=dict(repo=repo,subfolder=subfolder,api_http_status=response.status_code)
        if response.ok:
            data=response.json();revision=data['sha'];record['revision']=revision
            record['license']=data.get('cardData',{}).get('license')
            record['safetensors']=data.get('safetensors')
            record['files']=data.get('siblings',[])
            prefix=(subfolder+'/') if subfolder else ''
            config=get(f'https://huggingface.co/{repo}/resolve/{revision}/{prefix}config.json')
            record['config_http_status']=config.status_code
            if config.ok:record['config']=config.json()
            card=get(f'https://huggingface.co/{repo}/resolve/{revision}/README.md')
            if card.ok:record['model_card']=card.text
        candidates.append(record)
    upstream=[]
    for repo in ['SR-A-W/how-to-loop-moe','epfml/looped-moe','hed-ucas/LOOM',
                 'IQuestLab/loopie','ifm-ai/xllm-loop']:
        r=get('https://api.github.com/repos/'+repo+'/commits/HEAD')
        item=dict(repo=repo,http_status=r.status_code)
        if r.ok:
            revision=r.json()['sha'];item['revision']=revision
            card=get(f'https://raw.githubusercontent.com/{repo}/{revision}/README.md')
            item['README_http_status']=card.status_code
            if card.ok:item['README']=card.text
        upstream.append(item)
    def capture(args):return subprocess.check_output(args,text=True)
    environment=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        mask=MASK,physical_to_logical={str(g):r for r,g in enumerate(GPUS)},UUIDS=UUIDS,
        inventory=capture(['nvidia-smi']),topology=capture(['nvidia-smi','topo','-m']),
        git_parent=capture(['git','rev-parse','HEAD']).strip(),
        versions={name:importlib.metadata.version(name) for name in
                  ['torch','transformers','deep-ep','triton','einops']},
        spec_sha256=sha(ROOT/'authoritative_spec.md'),
        previous_research='poc_multibd_moe_ep/report.md',
        default_burner_mask='2,3,6,7; MUST override with VLLM_UTILIZE_GPUS=4,5,6,7')
    write(ROOT/'results/model_candidates.json',candidates)
    write(ROOT/'results/prior_source_audit.json',upstream)
    write(ROOT/'results/environment.json',environment)
    print([(c['repo'],c['api_http_status'],c.get('revision')) for c in candidates])


if __name__=='__main__':main()
