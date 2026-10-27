"""Capture source identities, hardware evidence, and verified historical counts."""
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from .common import ROOT, REPO, MODEL, DINFER, REFERENCE, LEGACY_EP, setup, write, digest, rows

def command(args, cwd=None):
    return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT)

def wilson(k,n):
    z=1.959963984540054
    p=k/n
    d=1+z*z/n
    c=(p+z*z/(2*n))/d
    w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-w,c+w]

def main():
    setup()
    evidence = dict(time=time.time(), gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,driver_version','--format=csv']),
        topology=command(['nvidia-smi','topo','-m']), cpu=command(['lscpu']), memory=command(['free','-h']),
        branch=command(['git','branch','--show-current'],REPO).strip(),
        git_sha=command(['git','rev-parse','HEAD'],REPO).strip(),
        dinfer_sha=command(['git','rev-parse','HEAD'],DINFER).strip(),
        reused_ep_sha=command(['git','rev-parse','HEAD'],LEGACY_EP).strip(),
        model_config=json.loads((MODEL/'config.json').read_text()))
    roots = [REFERENCE/'run_reference.py', REFERENCE/'score_reference.py', REFERENCE/'score_humaneval.py',
        REFERENCE/'sandbox_humaneval.sh', REFERENCE/'humaneval_sandbox_runner.py',
        LEGACY_EP/'poc_dllm_ep_comm/scripts/run_dllm_ep4.py', LEGACY_EP/'poc_dllm_ep_comm/instrumentation.py',
        DINFER/'python/dinfer/model/modeling_llada2_moe_sglang.py',
        DINFER/'python/dinfer/decoding/generate_uniform.py', DINFER/'python/dinfer/decoding/parallel_strategy.py']
    evidence['source_hashes']={str(p):digest(p) for p in roots}
    history={}
    for task,official in [('gsm8k',94.24),('humaneval',86.59)]:
        path=REFERENCE.parent/'artifacts/reproduction'/f'{task}_samples.jsonl'
        data=rows(path)
        field='correct' if task=='gsm8k' else 'humaneval_pass'
        k=sum(bool(r[field]) for r in data)
        history[task]=dict(kind='HISTORICAL_VERIFIED_NOT_NEW', path=str(path), sha256=digest(path),
            n=len(data), correct=k, accuracy_percent=100*k/len(data), wilson_95_percent=[100*x for x in wilson(k,len(data))],
            official_percent=official, difference_pp=100*k/len(data)-official,
            cap=sum(r['termination_reason']=='GEN_LENGTH_CAP' for r in data))
    evidence['historical_quality']=history
    write(ROOT/'environment.json',evidence)
    target=ROOT/'state/initial_git_index.json'
    if not target.exists():
        scopes=['.', ':!poc_flashvep/refinement_ep_allocation', ':!poc_flashvep/results/refinement_aware_ep_expert_allocation',
                ':!poc_flashvep/reports/0[0-9]_*.md', ':!poc_flashvep/reports/1[01]_*.md',
                ':!poc_flashvep/reports/refinement_aware_ep_expert_allocation_poc_spec.md']
        proc=subprocess.Popen(['git','ls-files','--stage','-z','--',*scopes], cwd=REPO,stdout=subprocess.PIPE)
        h=hashlib.sha256()
        while chunk:=proc.stdout.read(1024*1024):
            h.update(chunk)
        assert proc.wait()==0
        write(target,dict(sha256=h.hexdigest(), pathspec=scopes,time=time.time()))
    print(json.dumps(history,indent=2))

if __name__=='__main__':
    main()
