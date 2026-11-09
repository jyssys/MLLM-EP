"""Freeze held-out starts only after full-byte state and clean selection gates."""
from collections import defaultdict
from .common import *
from .predictive32 import prototype
from .qualification import qualify
from .analyze import metrics,percentile

SEEDS=(2026110911,2026110912,2026110913)


def qualified(model,family,diversity=False):
    for prefix in ('r10','r9','r8','r7','r6','r5','r4','r3','r1'):
        name=f'{prefix}_{model}_actual_{family}_state32'
        path=ROOT/'raw'/name/'dispatcher_result.json'
        if not path.exists() or not json.loads(path.read_text()).get('complete'):continue
        q=qualify(name,f'n32_{model}_native_full_state32')
        if (q.get('passed') and q.get('state',{}).get('exact') and q.get('async_handoff',{}).get('passed')
                and q.get('actual_dynamic_continuation_tested') and (not diversity or q['offload']['adaptive_diversity'])):
            return name
    return None


def calibrated(cfg,model):
    n=f'final_calibration_{model}_native128';z=f'final_calibration_{model}_zero128'
    native=metrics(n);zero=metrics(z);assert native['complete'] and zero['complete']
    folder=ROOT/'raw'/z;begin=json.loads((folder/'measurement_begin.json').read_text())['time']
    groups=defaultdict(list)
    for p in folder.glob('intervention_*.json'):
        for row in json.loads(p.read_text()).get('records',[]):
            if row['time']>=begin and row['event']=='DECODE_INTERVAL_SAMPLE':groups[row['batch']].append(row['interval_ms'])
    table={str(batch):percentile(values,.95)+1. for batch,values in groups.items() if len(values)>=8}
    assert table
    limit=min(native['frozen_SLO']['ITL_ms'],1.05*native['ITL_p99_ms'])
    cfg.update(native_ITL_target_ms=limit,feedback_ITL_limit_ms=limit,
        initial_decode_reserve_by_batch=table,decode_reserve_ms=native['ITL_p50_ms'],
        minimum_decode_reserve_ms=min(table.values()),
        native_calibration=dict(Original_Native=n,matched_zero=z,minimum_batch_samples=8,
            scope='Freeze the common estimator from independent N128 calibration controls before held-out starts. HBM pool matches the final comparison. Conditional p95 plus 1 ms is not a WCET bound.'))
    return cfg


def main():
    assert RESUMPTION is not None
    initialize();jobs=[];gates={};plan=[]
    for model in ('ouro','huginn'):
        token_gate=qualified(model,'token');loop_gate=qualified(model,'mixed',True)
        gates[model]=dict(token_state32=token_gate,actual_mixed_loop_state32=loop_gate)
        arms=['native','zero']
        if token_gate:arms.append('token')
        if loop_gate:arms.append('tile')
        # Whole-prompt expansion also needs actual accepted work at selection.
        full=metrics(f'r6_{model}_loop_selection32')
        full_q=qualify(f'r6_{model}_loop_selection32',f'r1_{model}_native_selection32')
        if loop_gate and full.get('Decode_SLO_screen') and full_q.get('offload',{}).get('adaptive_diversity'):arms.append('loop')
        for index,seed in enumerate(SEEDS):
            # Rotate execution order to avoid a fixed method/time confound.
            seed_arms=[arm for arm in arms if arm!='zero' or index==0]
            order=seed_arms[index%len(seed_arms):]+seed_arms[:index%len(seed_arms)]
            for arm in order:
                cfg=prototype(model,arm,count=128);cfg=calibrated(cfg,model)
                cfg.update(name=f'confirm_{model}_{arm}_seed{seed}',arrival_seed=seed,
                    empty_admission_uses_current_cap=True,
                    phase='HELD_OUT_ARRIVAL_CONFIRMATION',clean_confirmation=True,
                    state_probe=False,verify_transfer=False,timeout=600,startup_timeout=300,
                    comparison_reference=f'confirm_{model}_native_seed{seed}',
                    qualification_reference=token_gate if arm=='token' else loop_gate if arm in ('loop','tile') else None,
                    resumption_record='state/resumption_20261109.json')
                if arm not in ('native','zero'):
                    cfg['repeat_safety_references']=[f'confirm_{model}_{arm}_seed{previous}' for previous in SEEDS[:index]]
                if arm=='native':cfg.pop('comparison_reference')
                # Rebuild the Poisson offsets for this seed. Sampling seeds and texts remain paired.
                import random
                rng=random.Random(seed);offsets=[0.]
                for _ in range(127):offsets.append(offsets[-1]+rng.expovariate(cfg['arrival_rate']))
                cfg['arrival_offsets']=offsets
                save(cfg);jobs.append(cfg['name']);plan.append(dict(name=cfg['name'],model=model,arm=arm,seed=seed,
                    configuration_SHA256=digest(ROOT/'configs'/(cfg['name']+'.json'))))
    assert not (ROOT/'heldout_plan.json').exists(),'Do not retune held-out confirmations'
    write(ROOT/'heldout_plan.json',dict(created=time.time(),seeds=SEEDS,gates=gates,jobs=plan,
        text_scope='The 128 preserved text IDs are reused. Only arrival seeds are held out. No unseen-text claim.',
        zero_control_scope='One held-out zero control per model preserves time for all three Native, Token, and Loop starts. Additional matched zero controls belong to calibration or selection. No three-start held-out zero confidence interval is claimed.',
        rules='Keep all failures. Do not repeat an arm after an absolute Decode SLO violation. No tuning on these seeds.'))
    path=ROOT/'state/queue.json';queue=json.loads(path.read_text());write(path,queue+jobs)
    print(json.dumps(dict(gates=gates,jobs=jobs),indent=2))


if __name__=='__main__':main()
