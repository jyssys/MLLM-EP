"""Keep preregistered short-output diagnostics separate from the primary grid."""
import argparse
import random
import statistics
from .common import *
from .prepare import base
from .phase_b_configs import pd_config
from .followup_configs import save,enqueue


def tuned_g1():
    names=[]
    for model in ('ouro','huginn'):
        old=json.loads((ROOT/'configs'/f'phase_b_{model}_g1_screen_start0.json').read_text())
        c=dict(old,name=f'phase_b_{model}_g1_tuned_chunk_start0',chunk=2048,token_budget=2048,
            phase='focused_crossover',protocol_control='Native PD transport and identical arrivals. Only chunk budget changes.')
        names.append(save(c))
    enqueue(names)


def optional_capacity():
    for model in ('ouro','huginn'):
        for cell in ('g1','g2','g3','g4','g5','g6','c1','c2'):
            path=ROOT/'raw'/f'phase_b_{model}_{cell}_screen_start0'/'status.json'
            assert path.exists() and json.loads(path.read_text())['status']=='COMPLETE','Finish the full primary screen first'
    gate=json.loads((ROOT/'state/primary_phase_c_gate.json').read_text())
    assert not any(v.get('authorized') for v in gate['models'].values()),'Prioritize an eligible primary Phase C'
    names=[]
    for model,pool,slots,batches in (('ouro',22528,32,(1,8,16,24)),('huginn',26624,8,(1,2,3))):
        name=f'phase_a_{model}_short_output_diagnostic_start0'
        c=base(model,name)
        c.update(slots=slots,capacity=slots,chunk=2048,kv_blocks=pool,estimated_seconds=600,
            phase='short_output_diagnostic_capacity',
            diagnostic_scope='Preregistered 3500/32 controlled diagnostic. Does not replace the primary 128-256 output grid.',
            groups=[dict(chunk=2048,kv_blocks=pool,jobs=[dict(cell='D32',dataset='test_L3500.json',batch=b,cap=32,repeats=3) for b in batches])])
        names.append(save(c))
        c=pd_config(model,f'phase_a_{model}_short_output_prepared_pd_start0','D32',max(batches),100.)
        c.update(slots=slots,capacity=slots,chunk=2048,token_budget=2048,kv_blocks=pool,
            arrival_offsets=[0.]*c['count'],prepare_D_only=True,prepared_reference_run=name,
            phase='short_output_diagnostic_capacity',estimated_seconds=300)
        names.append(save(c))
    enqueue(names)


def optional_screen():
    from .b_capacity_plan import capacity
    names=[]
    for model,pool,slots in (('ouro',22528,32),('huginn',26624,8)):
        measured=capacity(model,'D32',2048,pool)
        path=ROOT/'raw'/f'phase_a_{model}_short_output_prepared_pd_start0'/'D_only_window.json'
        prepared=json.loads(path.read_text());assert prepared['native_output_parity']
        measured['mu_D']=min(measured['mu_D'],prepared['mu_D_finite'])
        measured['actual_prepared_PD_D']=prepared
        count=64;rate=.8*min(measured['mu_P'],measured['mu_D'])
        c=pd_config(model,f'phase_b_{model}_d32_diagnostic_start0','D32',count,rate)
        c.update(chunk=2048,token_budget=2048,slots=slots,capacity=slots,kv_blocks=pool,
            phase='focused_crossover',capacity_planning=measured,rate_factor=.8,estimated_seconds=600,
            diagnostic_scope='SECONDARY_SHORT_OUTPUT_CONTROLLED_DIAGNOSTIC_NOT_PRIMARY_GRID')
        names.append(save(c))
    enqueue(names)


def profile_expanded():
    old=json.loads((ROOT/'configs/phase_b_ouro_g3_expanded_start0.json').read_text())
    c=dict(old,name='profile_ouro_expanded_g3_v1',phase='profile',profile=True,
        profile_record_shapes=False,profile_schedule=dict(wait=200,active=400),estimated_seconds=600)
    enqueue([save(c)])


def alignment_smoke():
    names=[]
    for model,count,cell,pool,slots,reference,rate in (
        ('ouro',8,'G3',22528,32,'qualification_ouro_canonical24_reference_v1',2.520633344247786),
        ('huginn',4,'G2',26624,8,'phase_a_huginn_expanded_tuned_start0',.22)):
        for mode in ('together','staggered'):
            c=pd_config(model,f'diagnostic_{model}_prepared_{mode}_smoke_v1',cell,count,100.)
            rng=random.Random(2026103111);offsets=[0.]
            for _ in range(count-1):offsets.append(offsets[-1]+rng.expovariate(rate))
            c.update(slots=slots,capacity=slots,chunk=2048,token_budget=2048,kv_blocks=pool,
                arrival_offsets=[0.]*count,prepare_D_only=True,prepared_reference_run=reference,
                phase='batch_alignment_diagnostic',estimated_seconds=300,
                diagnostic_scope='All Prompt state is valid before Decode timing. The controlled D release pattern is not a Serving optimization.')
            if mode=='staggered':c['prepared_D_arrival_offsets']=offsets
            names.append(save(c))
    enqueue(names)


def alignment_finish():
    no_op=[]
    for model in ('ouro','huginn'):
        for mode in ('together','staggered','zero_offsets'):
            p=ROOT/'raw'/f'diagnostic_{model}_prepared_{mode}_smoke_v2'
            assert json.loads((p/'status.json').read_text())['status']=='COMPLETE'
            assert json.loads((p/'D_only_window.json').read_text())['native_output_parity']
        def tpot(mode):
            p=ROOT/'raw'/f'diagnostic_{model}_prepared_{mode}_smoke_v2'/'requests.jsonl'
            return statistics.median(r['tpot_ms'] for r in (json.loads(s) for s in p.read_text().splitlines()) if not r['warmup'])
        ratio=tpot('zero_offsets')/tpot('together')
        no_op.append(dict(model=model,zero_offset_TPOT_over_unmodified_gate=ratio,passes=ratio<=1.10))
    write(ROOT/'state/alignment_no_op_gate.json',dict(time=time.time(),rows=no_op,
        scope='Single-start instrumentation screen. This does not prove a small overhead bound.'))
    assert all(r['passes'] for r in no_op),'Do not expand a materially confounded alignment diagnostic'
    names=[]
    for mode in ('zero_offsets','staggered','together'):
        name=f'diagnostic_ouro_prepared_{mode}_smoke_v2'
        c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        base_cfg=pd_config('ouro',f'diagnostic_ouro_prepared_{mode}_b24_v2','G3',24,100.)
        c.update(name=base_cfg['name'],count=24,request_caps=base_cfg['request_caps'],prompt_ids=base_cfg['prompt_ids'],arrival_offsets=[0.]*24)
        if mode=='staggered':
            rng=random.Random(2026103111);offsets=[0.]
            for _ in range(23):offsets.append(offsets[-1]+rng.expovariate(2.520633344247786))
            c['prepared_D_arrival_offsets']=offsets
        elif mode=='zero_offsets':c['prepared_D_arrival_offsets']=[0.]*24
        names.append(save(c))
    enqueue(names)


def alignment_retry():
    names=[]
    for model in ('ouro','huginn'):
        for mode in ('together','staggered','zero_offsets'):
            source_mode='together' if mode=='zero_offsets' else mode
            source=f'diagnostic_{model}_prepared_{source_mode}_smoke_v1'
            c=json.loads((ROOT/'configs'/(source+'.json')).read_text())
            c.update(name=f'diagnostic_{model}_prepared_{mode}_smoke_v2',alignment_variant=mode,
                diagnostic_note='Configuration is cached outside the scheduling callback. Zero offsets measure the gate path without release delay.')
            if mode=='zero_offsets':c['prepared_D_arrival_offsets']=[0.]*c['count']
            names.append(save(c))
    write(ROOT/'state/alignment_v1_measurement_exclusion.json',dict(time=time.time(),
        excluded_from_alignment_performance_comparison=[f'diagnostic_{m}_prepared_{k}_smoke_v1' for m in ('ouro','huginn') for k in ('together','staggered')],
        reason='The diagnostic callback read configuration files repeatedly. The V1 outputs remain valid correctness evidence, but performance attribution is confounded.',
        hypotheses=['Configuration file IO inside the scheduling callback','Intended release staggering changes batch packing','Independent runtime variation'],
        red_command='python -m unittest poc_flashvep.looped_pd_bottleneck_crossover.test_contracts.ContractTests.test_prepared_gate_has_no_configuration_IO -v',
        red_result='FAIL: three configuration-read call sites exist inside the callback',
        scope='Primary Serving runs do not install this prepare_D_only callback and are not affected.'))
    enqueue(names)


def main():
    actions=dict(tuned_g1=tuned_g1,optional_capacity=optional_capacity,optional_screen=optional_screen,profile_expanded=profile_expanded,
        alignment_smoke=alignment_smoke,alignment_finish=alignment_finish,alignment_retry=alignment_retry)
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=tuple(actions));actions[parser.parse_args().action]()


if __name__=='__main__':main()
