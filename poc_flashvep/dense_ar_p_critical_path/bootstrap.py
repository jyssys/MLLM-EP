"""Rebase only the evidence root and keep the original strong baselines."""
import hashlib
import json
import subprocess
import time
import zipfile
from .common import ROOT,PRIOR,REPO,PYTHON,write,digest,copy_data,initialize,bind,preserve


def reroot(source_name):
    source=PRIOR/'references/source_capsules'/source_name
    target=ROOT/'references/source_capsules'/('pc_'+source_name)
    if target.exists():return target
    with zipfile.ZipFile(source) as archive:files={name:archive.read(name) for name in archive.namelist()}
    key='poc_flashvep/dense_ar_longbench400_pd/common.py';before=files[key]
    needle="ROOT=REPO/'poc_flashvep/reports/dense_ar_longbench400_v2_20261122'"
    assert before.decode().count(needle)==1
    files[key]=before.decode().replace(needle,"ROOT=REPO/'poc_flashvep/reports/dense_ar_p_critical_path_20261123'").encode()
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(item,data)
    write(target.with_suffix('.json'),dict(time=time.time(),source=str(source.relative_to(REPO)),source_SHA256=digest(source),
        SHA256=digest(target),changed=[key+':ROOT'],all_other_worker_bytes_identical=True,
        files={name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        scope='Only the evidence ROOT literal changed. No model,scheduler,compiler,transport or numerical operator changed. '
            'Inherited state/quality proofs are labelled inherited,not fresh GPU tests. New runtime interventions require fresh gates.'))
    return target


def baseline(label,name,pair,seed=None,natural=False,save_cfg=True):
    runner=bind()
    original=PRIOR/'configs'/f'V2_C3_Heldout192Replay_q2.0_s0_{label}.json'
    cfg=json.loads(original.read_text());old_gate=PRIOR/'state'/cfg['qualification_gate_v2']
    proof=json.loads(old_gate.read_text());assert proof['passed'] and proof['full400_passed'] and proof['quality_passed']
    assert runner.signature(cfg)==proof['signature']
    archive=reroot(cfg['source_capsule']);copy_data(cfg['dataset'])
    cfg.update(name=name,gpus=list(pair),source_capsule=archive.name,source_capsule_SHA256=digest(archive))
    if seed is not None:
        import numpy as np
        cfg.update(seed=seed,arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/cfg['qps'],cfg['count']-1))].tolist())
    if natural:
        cfg.pop('replay_output_lengths',None);cfg.pop('diagnostic_controlled_output_replay',None);cfg['ignore_eos']=False
    gate='pc_inherited_'+label+'_root_only.json';cfg['qualification_gate_v2']=gate
    assert runner.signature(cfg)==proof['signature']
    record=dict(time=time.time(),passed=True,full400_passed=True,quality_passed=True,
        signature=runner.signature(cfg),source_capsule_SHA256=digest(archive),evidence='INHERITED_CORRECTNESS_ROOT_ONLY',
        inherited_gate=str(old_gate.relative_to(REPO)),inherited_gate_SHA256=digest(old_gate),
        root_equivalence_proof_SHA256=digest(archive.with_suffix('.json')),
        full400_scope='Inherited full400 proof. This is not a new full400 execution.',
        scope='The baseline numerical and runtime code is byte-identical except the evidence ROOT literal. '
            'The strong original compiled Token is retained. No intervention flag is covered by this root-only proof.')
    path=ROOT/'state'/gate
    if path.exists():
        old=json.loads(path.read_text());assert old['signature']==record['signature'] and old['source_capsule_SHA256']==record['source_capsule_SHA256']
    else:write(path,record)
    if save_cfg:runner.save(cfg)
    return cfg


def launch(plan_name):
    plan=json.loads((ROOT/'jobs'/plan_name).read_text());children=[]
    for lane,jobs in enumerate(plan['lanes']):
        if not jobs:continue
        log=ROOT/'logs'/(plan_name.removesuffix('.json')+f'_lane{lane}.log');assert not log.exists()
        with log.open('w') as stream:
            child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.runner',plan_name,str(lane)],
                cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(dict(lane=lane,pid=child.pid,log=str(log.relative_to(ROOT))))
    write(ROOT/'state'/(plan_name.removesuffix('.json')+'_controller.json'),dict(time=time.time(),children=children))
    return children


def prepare():
    initialize();bind();preserve()
    # Replay the published trace on its original pair before changing a runtime factor.
    names=[];lanes=[[],[],[],[]]
    for label in ('NativeStrong','NativeCommon2','NativeCommon0','Layer','StockToken','Hybrid'):
        name='PC_R0_Replay192_q2_s0_'+label;baseline(label,name,[4,5]);lanes[2].append(name);names.append(dict(label=label,name=name))
    target=ROOT/'jobs/baseline_reproduction.json';assert not target.exists()
    write(target,dict(time=time.time(),lanes=lanes,arms=names,stop_on_failure=True,
        scope='Original heldout192,per-request compiled-C1 output lengths,arrival seed2026112281,QPS2 and GPU4/5. '
            'This first start is a reproduction/diagnostic,not new N3 confirmation. Do not pool it with historical starts.'))
    return launch(target.name)


if __name__=='__main__':print(prepare(),flush=True)
