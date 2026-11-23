"""Use Discovery208,not heldout timing,to screen two geometry controls."""
import json
import subprocess
import time
import numpy as np
from .common import ROOT,PRIOR,REPO,PYTHON,write,digest,bind,dispatch_allowed
from .bootstrap import baseline


def prepare():
    runner=bind()
    from ..dense_ar_output_aware_v2.work_contract import regenerate
    data_name='balanced400_v2_discovery.json';data=json.loads((ROOT/'data'/data_name).read_text());n=len(data)
    reference=json.loads((PRIOR/'state/official_compiled_c1_reference_complete_v2.json').read_text())
    source=PRIOR/'raw'/reference['name']/'requests.jsonl'
    refs={r['prompt_id']:r['output_tokens'] for r in map(json.loads,source.read_text().splitlines())}
    lengths=[refs[r['prompt_id']] for r in data];seed=2026112341;qps=2.
    rows=[];jobs=[]
    for label in ('NativeStrong','NativeCommon2','StockToken','OriginalLayer','Layer8','OriginalHybrid','HybridIso8Approx'):
        name='PC_BalanceDiscovery208_'+label
        if label in ('Layer8','HybridIso8Approx'):
            cfg=json.loads((ROOT/'configs'/f'PC_Balance_{label}_Natural400.json').read_text())
            cfg['qualification_gate_v2']='pc_balance_'+label.lower()+'_composition.json'
        elif label.startswith('Original'):
            cfg=json.loads((ROOT/'configs'/f'PC_C3_EqualReplay_s0_{label}.json').read_text())
        else:cfg=baseline(label,name,[6,7],save_cfg=False)
        cfg.update(name=name,gpus=[6,7],dataset=data_name,dataset_SHA256=digest(ROOT/'data'/data_name),count=n,
            qps=qps,seed=seed,arrival_offsets=np.r_[0.,np.cumsum(np.random.default_rng(seed).exponential(1/qps,n-1))].tolist(),
            ignore_eos=True,output_cap=2048,diagnostic_controlled_output_replay=True,replay_output_lengths=lengths,
            correctness_only=False,axis_state_capture=False,state_capture=False,gpu_state_digest=False,
            numerical_probe=False,numerical_probe_sync=False,kernel_activity=False,serial=False,
            discovery_only=True,confirmation=False)
        cfg.pop('request_holdout_scope',None);regenerate(cfg,data);runner.save(cfg)
        jobs.append(name);rows.append(dict(label=label,name=name,gate=cfg['qualification_gate_v2']))
    path=ROOT/'jobs/p_balance_discovery208_v1.json';assert not path.exists()
    write(path,dict(time=time.time(),lanes=[[],[],[],jobs],arms=rows,stop_on_failure=True,
        dataset_SHA256=digest(ROOT/'data'/data_name),reference_SHA256=digest(source),seed=seed,qps=qps,
        screening=dict(minimum_absolute_decode_attainment=.95,
            selection='Confirm a candidate if its same-axis P critical-path ratio≤.95 OR its TTFT/strong-Stock point ratio≤1.05. '
                'The1.05 point screen is not a success threshold. Final TTFT CI upper must be<1 and decode CI upper≤1.05.'),
        scope='Two predefined geometry candidates only. All208 discovery requests remain frozen. '
            'Same independent-C1 output work per request,same trace and pair for all seven arms. '
            'Keep both favorable and unfavorable candidates. Do not claim an axis win from one start. '
            'The original best-tuned Stock cap0/credit0 stays intact. Original same-axis controls retain credit16/8 and cap2. '
            'Not a common-policy or exact-attention-work comparison. Record actual aligned GEMM rows.'))
    log=ROOT/'logs/p_balance_discovery208_v1_controller.log';assert not log.exists()
    with log.open('w') as stream:
        child=subprocess.Popen([str(PYTHON),'-u','-m','poc_flashvep.dense_ar_p_critical_path.balance_discovery','run'],
            cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    write(ROOT/'state/p_balance_discovery_controller.json',dict(time=time.time(),pid=child.pid))
    return child.pid


def run():
    runner=bind()
    for label in ('layer8','hybridiso8approx'):
        path=ROOT/'state'/f'pc_balance_{label}_composition.json'
        while not path.exists():assert dispatch_allowed();time.sleep(10)
        assert json.loads(path.read_text())['passed'],path
    runner.lane('p_balance_discovery208_v1.json',3)


if __name__=='__main__':
    import sys
    print(prepare() if sys.argv[1]=='prepare' else run(),flush=True)
