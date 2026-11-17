"""Select only on discovery. Freeze before unseen candidate latency runs."""
import collections
import fcntl
import json
import random
import statistics
import time
from .common import ROOT,SOURCE,QUIET_LOCK,resources,write,save,digest
from .analysis import metrics
from .run import execute
from .heldout_baselines import SEEDS
from .fresh_reference import REFERENCE
from ..duetserve_agg_decode.configuration import offsets

WORKERS=('common.py','site_bootstrap.py','serving.py','dynamic_serving.py',
    'token_axis.py','credit_fast_lane.py','flow_policy.py','replay.py','recording.py','geometry_control.py')
FINGERPRINT=('entry_module','token_axis','token_suffix','static_layers','migration_fraction',
    'P_group_size','P_slots','D_admission_batch','D_slots','D_wait_for_source_macro',
    'D_combine_ready_macros','preplanned_source_groups','D_token_budget','D_prefill_chunk',
    'ar_cudagraph_mode','ar_async_scheduling','ar_compile_mode','async_decode','async_ingress',
    'hidden_source_slots','hidden_destination_slots','flow_balance_policy','flow_horizon_groups',
    'flow_cost_model','direct_phase_norm','gpu_phase_mask_norm','source_token_release_policy','flow_adaptive_source')


def wait_file(path, count=None):
    heartbeat=0.
    while not path.exists() or (count is not None and len(json.loads(path.read_text()))<count):
        if time.time()>json.loads((ROOT/'state/progress.json').read_text())['deadline_epoch']-7200:
            raise TimeoutError('Reserve confirmation and closeout time: '+str(path))
        if time.time()-heartbeat>=60:
            heartbeat=time.time();print('WAIT_DISCOVERY_DEPENDENCY',path.name,count,flush=True)
        time.sleep(15)


def train_records(cap):
    result=[]
    for path in sorted((ROOT/'raw').glob('*/dispatcher_result.json')):
        if not json.loads(path.read_text()).get('complete'):
            continue
        cfg=json.loads((path.parent/'configuration.json').read_text())
        if cfg.get('correctness_only') or cfg.get('execution_events') or cfg.get('boundary_events') or cfg.get('credit_fast_lane'):
            continue
        if 'heldout' in cfg['dataset'].lower() or cfg.get('phase','').startswith('FROZEN_HELDOUT'):
            continue
        if cfg['count']!=(128 if cap==16 else 64) or set(cfg['request_caps'])!={cap}:
            continue
        if abs(cfg.get('arrival_rate',0)-(6.4 if cap==16 else 4.8))>1e-6:
            continue
        method=cfg.get('evaluation_arm')
        if method not in ('Token','Static','Dynamic'):
            continue
        value=metrics(cfg['name'])
        assert value['output_gate']['passed']
        result.append((cfg,value))
    return result


def choose(cap):
    native=f'train_Native_N128_O16_q6.4_v1' if cap==16 else 'train_Native_N64_O64_q4.8_v1'
    wait_file(ROOT/'raw'/native/'dispatcher_result.json')
    baseline=metrics(native)
    candidates=collections.defaultdict(list)
    for cfg,value in train_records(cap):
        key=(cfg['evaluation_arm'],json.dumps({k:cfg.get(k) for k in FINGERPRINT},sort_keys=True))
        candidates[key].append((cfg,value))
    selected={};frontier=[]
    for method in ('Token','Static','Dynamic'):
        rows=[]
        for (kind,key),items in candidates.items():
            if kind!=method:
                continue
            safe=all(value['ITL_p99_ms']<=1.05*baseline['ITL_p99_ms'] and
                value['TPOT_p95_ms']<=1.05*baseline['TPOT_p95_ms'] and
                value['SLO_regimes']['Decode_priority']['percentile_Decode_absolute_pass']
                for _,value in items)
            score=statistics.mean(value['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput'] for _,value in items)
            ttft=statistics.mean(value['TTFT_p95_ms'] for _,value in items)
            rows.append(dict(method=method,representative=items[0][0],
                discovery_names=[cfg['name'] for cfg,_ in items],starts=len(items),safe_point_screen=safe,
                mean_SLO_tokens_s=score,mean_TTFT_p95_ms=ttft))
        assert rows,'No discovery candidate: '+method
        legal=[row for row in rows if row['safe_point_screen']]
        best=max(legal or rows,key=lambda row:(row['mean_SLO_tokens_s'],-row['mean_TTFT_p95_ms']))
        selected[method]=best
        frontier.extend(rows)
    contract=dict(cap=cap,selected=selected,frontier=frontier,
        native_discovery=baseline['name'],selection_objective='Decode-priority qualified output tokens/s. Discovery point safety screen, not a noninferiority proof.',
        SLO_SHA256=digest(ROOT/'configs/frozen_SLO_regimes.json'),
        dataset_SHA256=digest(ROOT/'data/heldout128_8000.json'),
        worker_hashes={name:digest(SOURCE/name) for name in WORKERS},
        primary_QPS=6.4 if cap==16 else 4.8,seeds=list(SEEDS),
        scope='Each execution configuration pools all matching discovery starts. No single best noisy replicate is selected. Candidate selection cannot read unseen latency. Main relative gates use paired independent starts. Other QPS can have only one exploratory start.')
    path=ROOT/'state'/f'candidate_confirmation_freeze_O{cap}.json'
    if path.exists():
        assert json.loads(path.read_text())['contract']==contract
    else:
        write(path,dict(time=time.time(),contract=contract))
    print('FROZEN',cap,{m:r['representative']['name'] for m,r in selected.items()},flush=True)
    return contract


def qualify_suffix(suffix):
    known=ROOT/'analysis'/f'unseen_Token{suffix}_axis_state_N128_O16_hash_v1_hash_state_gate.json'
    if known.exists() and json.loads(known.read_text())['passed']:
        return
    from .qualification import configuration
    from .hash_qualification import gate
    source=f'selected_Native_partition{suffix}_N128_O2_v1'
    candidate=f'selected_Token{suffix}_state_N128_O2_v1'
    for name,actual in ((source,None),(candidate,suffix)):
        cfg=configuration(name,count=128,suffix=actual)
        cfg.update(dataset='heldout128_8000.json',reference_name=REFERENCE,cap=2,request_caps=[2]*128,
            axis_state_save_tensors=False,witness_hidden_partition_tokens=[suffix])
        assert execute(save(cfg))
    gate(candidate,source)


def config(contract,method,qps,seed):
    cap=contract['cap'];cfg=dict(contract['selected'][method]['representative'])
    cfg.update(name=f'heldout_{method}_N128_O{cap}_q{qps:g}_s{seed}_v1',
        count=128,cap=cap,request_caps=[cap]*128,dataset='heldout128_8000.json',
        reference_name=REFERENCE,arrival_seed=seed,arrival_rate=qps,
        arrival_offsets=offsets(128,qps,seed),phase='FROZEN_HELDOUT_CANDIDATE',
        confirmation_method=method,correctness_only=False,clean_timing=True,
        prompt_reuse_scope='128 unique unseen source documents. No prefix caching.')
    return cfg


def main():
    values=[]
    try:
        for cap in (16,64):
            if cap==16:
                wait_file(ROOT/'analysis/frontier_followup.json',9)
                wait_file(ROOT/'analysis/macro8_axis_frontier.json',4)
                wait_file(ROOT/'analysis/adaptive_token_fine_frontier.json',4)
                wait_file(ROOT/'analysis/flow4_discovery_screen.json',2)
                wait_file(ROOT/'analysis/adaptive_source_flow_discovery.json',2)
            else:
                wait_file(ROOT/'analysis/output64_axis_frontier.json',12)
                wait_file(ROOT/'analysis/flow_homogeneous64_discovery.json',2)
                wait_file(ROOT/'analysis/adaptive_output64_flow_discovery.json',2)
                wait_file(ROOT/'analysis/output64_fraction_token_refinement.json',5)
            contract=choose(cap)
            qualify_suffix(contract['selected']['Token']['representative']['token_suffix'])
            # The primary comparison receives three independent starts first.
            # Neighboring QPS are explicitly exploratory until replicated.
            schedule=[(6.4 if cap==16 else 4.8,seed) for seed in SEEDS]
            if cap==64:
                schedule.extend((6.4,seed) for seed in SEEDS)
            schedule.extend((qps,SEEDS[0]) for qps in (2.,10.,4.8) if qps not in {q for q,_ in schedule})
            for qps,seed in schedule:
                methods=['Token','Static','Dynamic'];random.Random(seed).shuffle(methods)
                for method in methods:
                    for name,expected in contract['worker_hashes'].items():
                        assert digest(SOURCE/name)==expected,'Frozen worker source changed: '+name
                    cfg=config(contract,method,qps,seed)
                    if not execute(save(cfg)):
                        continue
                    value=metrics(cfg['name']);assert value['output_gate']['passed']
                    values.append(value);write(ROOT/'analysis/heldout_candidate_results.json',values)
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():
                resources().restore(list(range(8)))


if __name__=='__main__':
    main()
