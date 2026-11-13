"""Evaluate small real online decisions after independent cost calibration."""
import fcntl
from .common import *
from .prepare import base
from .qualify import dynamic_config
from .run import execute
from .analyze import metrics,compare,records


def serving_config(name,cap=32,count=64,rate=4.777100014491616,seed=2026111301,dataset='discovery.json',**kwargs):
    c=dynamic_config(name,count=8)
    b=base(name,count=count,cap=cap,rate=rate,seed=seed,dataset=dataset)
    c.update({k:b[k] for k in ('count','cap','dataset','request_caps','arrival_offsets','arrival_rate','arrival_seed')})
    c.update(kv_layout='HND',correctness_only=False,numerical_native_reference=False,
        hidden_verify_hash=False,verify_native_prefix_KV=False,verify_layer_execution=False,execution_events=False,
        P_group_size=4,D_admission_batch=16,hidden_destination_slots=16,
        policy='COST',cost_model='calibrated_cost_model.json',minimum_P_backlog=1,
        safety_margin_ms=5.,minimum_predicted_benefit_ms=0.,maximum_state_age_ms=250.,
        decode_deadline_ms=34.,phase='ACTUAL_ONLINE_POLICY_DISCOVERY_NOT_HELDOUT_CONFIRMATION')
    c.update(kwargs);return c


def main():
    initialize();results=[]
    try:
        qualification=[]
        for mode in ('COST','DEADLINE','IDLE'):
            c=serving_config(f'actual_online_{mode}_N32_qualification_v1',cap=16,count=32,
                policy=mode,verify_layer_execution=True,phase='ACTUAL_OBSERVED_STATE_DYNAMIC_N32_QUALIFICATION')
            if not execute(save(c),restore=False):return
            ref=next(p.stem for p in (ROOT/'configs').glob('coarse_O16_N64_*native16_s1_v1.json'))
            gate=compare(ref,c['name'],True)
            commits=records(ROOT/'raw'/c['name']/'actual_cut_commits.jsonl')
            actual={r['offloaded_layers'] for r in commits}
            row=dict(name=c['name'],output_gate=gate,actual_layers=sorted(actual),
                actual_online_state_adaptation='DEMONSTRATED' if any(actual) and len(actual)>1 else 'ADAPTIVE_NOT_DEMONSTRATED',
                forced=False,scope='Actual observed state and frozen costs. A single positive cut or all-Native outcome is not variable-cut proof.')
            qualification.append(row);write(ROOT/'state/actual_dynamic_N32_qualification.json',dict(results=qualification))
            print('ACTUAL_DYNAMIC_N32',mode,gate['passed'],sorted(actual),flush=True)
            if not gate['passed']:return
        for cap in (16,32,64):
            ref=next(p.stem for p in (ROOT/'configs').glob(f'coarse_O{cap}_N64_*native16_s1_v1.json'))
            for mode in ('NATIVE','COST','DEADLINE','IDLE'):
                c=serving_config(f'online_{mode}_O{cap}_N64_v1',cap=cap,policy=mode)
                if not execute(save(c),restore=False):return
                gate=compare(ref,c['name']);commits=records(ROOT/'raw'/c['name']/'actual_cut_commits.jsonl')
                counts={str(n):sum(len(r['requests']) for r in commits if r['offloaded_layers']==n) for n in (0,2,4,8,12)}
                row=dict(name=c['name'],output_gate=gate,metrics=metrics(c['name']),actual_layer_distribution=counts,
                    reservation_lifetime=json.loads((ROOT/'raw'/c['name']/'reservation_lifetime_decode.json').read_text()))
                results.append(row);write(ROOT/'state/online_pilot_progress.json',dict(results=results))
                print('ONLINE_PILOT',c['name'],gate['passed'],counts,row['metrics']['TTFT_p95_ms'],row['metrics']['ITL_p99_ms'],flush=True)
                if not gate['passed']:return
        write(ROOT/'analysis/online_pilot_results.json',dict(results=results,
            scope='Actual online decisions use past observations and a frozen primitive model. These discovery results do not establish independent confirmation or non-inferiority.'))
    finally:
        with QUIET_LOCK.open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
