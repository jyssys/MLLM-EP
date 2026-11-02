"""Freeze preservation, data, thresholds, and one-variable recovery controls."""
import argparse
import copy
import gzip
import random
import subprocess
from .common import *


def put(cfg, n, rate=None, split='test'):
    path = ROOT/'data'/f'{split}_{cfg["model"]}_L{cfg["length"]}.json'
    rows = json.loads(path.read_text())
    assert n <= len(rows), 'Do not silently cycle prompts'
    cfg.update(count=n, dataset=path.name, request_caps=[cfg['cap']]*n,
        prompt_ids=[r['prompt_id'] for r in rows[:n]], arrival_offsets=[0.]*n)
    if rate is not None:
        rng=random.Random(cfg['arrival_seed'])
        for i in range(1,n):
            cfg['arrival_offsets'][i]=cfg['arrival_offsets'][i-1]+rng.expovariate(rate)
        cfg['arrival_rate']=rate
        cfg['arrival_pattern']='OPEN_LOOP_POISSON'
    else:
        cfg.pop('arrival_rate',None)
        cfg['arrival_pattern']='SERIAL_CALIBRATION' if cfg.get('calibration_serial') else 'SIMULTANEOUS_VALIDATION_OR_CAPACITY'
    cfg['atomic_prompt']=cfg['chunk'] >= cfg['length']
    cfg['chunk_label']='full' if cfg['atomic_prompt'] else str(cfg['chunk'])
    target=ROOT/'configs'/(cfg['name']+'.json')
    if target.exists():
        assert json.loads(target.read_text()) == cfg, 'Immutable configuration conflict'
    else:
        write(target,cfg)
    return cfg['name']


def initial():
    setup()
    initial=ROOT/'state/preservation.json'
    if not initial.exists():
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
        assert head == 'f4ecea59f36f620c69c4fef09c590bf790802cd9'
        signature=index_signature()
        changes=subprocess.check_output(['git','diff','--name-status','-z'],cwd=REPO)
        with gzip.open(ROOT/'state/original_worktree_changes.nul.gz','wb') as handle:
            handle.write(changes)
        (ROOT/'state/private_runtime.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=RLT))
        write(initial,dict(time=time.time(),base=head,index_entries_SHA256=signature,
            private_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=RLT,text=True).strip(),
            private_diff_SHA256=digest(ROOT/'state/private_runtime.patch'),
            worktree_status_SHA256=hashlib.sha256(changes).hexdigest()))
        subprocess.run(['git','switch','-c',BRANCH],cwd=REPO,check=True)
        assert index_signature()==signature
        write(ROOT/'state/start.json',dict(time=time.time(),scope='New campaign. No inherited deadline or CLOSED state.'))
    preserve()
    corpus=json.loads((PRIOR/'data/long_source_pool.json').read_text())
    lengths={'ouro':[512,4096,6144,8192], 'huginn':[512,2048,2560,3072,3500]}
    for model,values in lengths.items():
        for length in values:
            for split,pool in (('calibration',corpus[:24]),('test',corpus[24:])):
                rows=[dict(prompt_id=r['prompt_id'],source_SHA256=r['source_SHA256'],
                    source_dataset=r['source_dataset'],source_type='LENGTH_MATCHED_CONTROLLED_PUBLIC_TEXT_PREFIX',
                    truncation_explicit=True,input_ids={model:r['input_ids'][model][:length]},length=length)
                    for r in pool if len(r['input_ids'][model])>=length]
                assert len(rows)>=16
                path=ROOT/'data'/f'{split}_{model}_L{length}.json'
                if path.exists():assert json.loads(path.read_text())==rows
                else:write(path,rows)
    write(ROOT/'state/data_provenance.json',dict(source=str(PRIOR/'data/long_source_pool.json'),
        SHA256=digest(PRIOR/'data/long_source_pool.json'),calibration_sources=24,test_sources=len(corpus)-24,
        prior_provenance=str(PRIOR/'configs/dataset_provenance.json'),
        scope='Pinned model-tokenizer prefixes from the previous corpus. Controlled lengths and ignore_eos. Not Azure or natural EOS.'))
    old=configure()
    controls=[]
    seed_cfg=json.loads((PRIOR/'configs/pilot_ouro_l8192_o64_cfull_f1p0_v1.json').read_text())
    for label,delta in [('original',{}),('workspace128',{'D_token_budget':128}),
        ('workspace128_pool25088',{'D_token_budget':128,'D_kv_blocks':25088}),
        ('workspace128_pool25088_drain0',{'D_token_budget':128,'D_kv_blocks':25088,
            'native_transport_limits':{'max_draining_requests':0}})]:
        gate=copy.deepcopy(seed_cfg)
        gate.update(name='recovery_ouro_8192_64_'+label+'_gate8',phase='correctness',verify_transfer=True,state_probe=True,
            timeout=300,gpus=[0,1],nsys=False,worker='rlt_worker',**delta)
        put(gate,8)
        smoke=copy.deepcopy(gate)
        smoke.update(name='recovery_ouro_8192_64_'+label+'_pilot24',phase='qps_pilot',verify_transfer=False,state_probe=False,timeout=300)
        put(smoke,24,seed_cfg['arrival_rate'])
        controls.append(dict(label=label,gate=gate['name'],pilot=smoke['name']))
    h=json.loads((PRIOR/'configs/ext_final_huginn_l3500_c2048_r1_s1_v1.json').read_text())
    h.update(name='recovery_huginn_3500_32_control_gate8',phase='correctness',count=8,verify_transfer=True,state_probe=True,
        gpus=[4,5],nsys=False,worker='rlt_worker',timeout=300)
    put(h,8)
    profile=[]
    # The first comparison changes no profiling setting. Its partner launches
    # only P under Nsight and captures a bounded 3-second device window.
    for n in (1,4,8,24):
        a=copy.deepcopy(h)
        a.update(name=f'recovery_huginn_3500_32_clean{n}',phase='profile_control',verify_transfer=False,state_probe=False,timeout=180)
        put(a,n,.9689)
        b=copy.deepcopy(a)
        b.update(name=f'recovery_huginn_3500_32_p_only_profile{n}',P_only_profiler=True,
            capture_seconds=3.,capture_after_submissions=2,nsys=True)
        # nsys=True supplies the existing forward NVTX annotations. The runner
        # does not wrap the parent process or the D child in Nsight.
        put(b,n,.9689)
        profile.append(dict(clean=a['name'],profile=b['name']))
    write(ROOT/'state/initial_plan.json',dict(ouro_controls=controls,huginn_gate=h['name'],huginn_profiles=profile))
    write(ROOT/'state/preregistered_gates.json',dict(time=time.time(),profile_wall_difference_limit=.05,
        P_kernel_busy_min=.8,P_service_reference_fraction_min=.85,D_TPOT_limit_fraction=.9,
        final_requests_min=128,positive_independent_starts_min=3,
        same_start_same_window_required=True,HF_original_fidelity='NOT_ESTABLISHED',
        original_length_fixes_first=True,model_lengths=lengths))
    print(json.dumps(dict(status='PREPARED',branch=BRANCH,root=str(ROOT),ouro_controls=controls,huginn_profile_pairs=len(profile))))


if __name__=='__main__':
    initial()
