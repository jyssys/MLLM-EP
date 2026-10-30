"""Model-specific release gate. No GPU model loading occurs here."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .common import ROOT,CF_ROOT,CF_COMMIT,REPO,MODELS,MODEL_PINS,VARIANTS,RLT,setup,write,digest


def main():
    setup()
    heads=subprocess.check_output(['git','ls-remote','--heads','origin','flashvep/looped-pd-b-numerical-diagnosis'],cwd=REPO,text=True)
    if not heads.startswith(CF_COMMIT+'\t'):raise RuntimeError('The numerical release remote SHA does not match')
    critical=['poc_flashvep/looped_pd_b_numerical_diagnosis/patches.py',
              'poc_flashvep/looped_pd_b_numerical_diagnosis/norm_probe.py']
    hashes={}
    for name in critical:
        expected=subprocess.check_output(['git','show',CF_COMMIT+':'+name],cwd=REPO)
        actual=(REPO/name).read_bytes()
        if expected!=actual:raise RuntimeError('The qualified numerical implementation changed: '+name)
        hashes[name]=hashlib.sha256(actual).hexdigest()
    locked=json.loads((CF_ROOT/'configs/read_only_source_lock.json').read_text())
    changed=[p for p,sha in locked.items() if not Path(p).exists() or digest(p)!=sha]
    if changed:raise RuntimeError('An immutable original engine source changed')
    active=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try:
            command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if proc.stat().st_uid==os.getuid() and any((' -m poc_flashvep.looped_pd_b_numerical_diagnosis.'+m+' ') in command+' '
                for m in ('worker','adapter_qualification','release_checks')):active.append(int(proc.name))
        except (OSError,UnicodeError):continue
    if active:raise RuntimeError('The numerical task still owns active experiment processes: '+str(active))
    results={}
    for model in ('ouro','huginn'):
        source=CF_ROOT/'state'/('ouro_down_norm_qualification.json' if model=='ouro' else 'huginn_adapter_qualification.json')
        value=json.loads(source.read_text())
        passed=value['status']=='FINITE_QUALIFICATION_PASS'
        for check in value['checks']:
            compared=check.get('comparisons',[])
            passed=passed and all(r['exact']==r['N']==check.get('count') and r['complete'] for r in compared)
        if model=='huginn':
            release=json.loads((CF_ROOT/'state/final_release_checks.json').read_text())
            passed=passed and release['status']=='INTERNAL_CORRECTNESS_RELEASE_PASS'
        config=json.loads((MODELS[model]/'config.json').read_text())
        results[model]=dict(status='CORRECTNESS_READY' if passed else 'BLOCKED_CORRECTNESS',variant=VARIANTS[model],
            source_verdict=str(source.relative_to(REPO)),checkpoint=MODEL_PINS[model],
            context_limit=config.get('max_position_embeddings',config.get('block_size')),
            original_HF_equivalence='NOT_ESTABLISHED',benchmark_answer_accuracy='NOT_MEASURED',
            scope='Matched-runtime fixed-depth output and state qualification only. Every RCC mechanism requires a new gate.')
    rlt_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RLT,text=True).strip()
    diff=subprocess.check_output(['git','diff','--binary'],cwd=RLT)
    record=dict(correctness_commit=CF_COMMIT,remote_SHA_verified=True,models=results,
        critical_source_sha256=hashes,read_only_source_files_checked=len(locked),old_experiment_processes=active,
        RLT_HEAD=rlt_head,RLT_preserved_local_diff_sha256=hashlib.sha256(diff).hexdigest(),
        GPU_topology=subprocess.check_output(['nvidia-smi','topo','-m'],text=True),
        GPU_UUIDs=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True),
        spec_sha256=digest(REPO/'poc_flashvep/reports/rcc_pd_cross_depth_cobatching_poc_spec.md'))
    write(ROOT/'configs/prerequisite_gate.json',record)
    print({m:r['status'] for m,r in results.items()})


if __name__=='__main__':main()
