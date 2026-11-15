"""Freeze selected methods before independent unseen-prompt evaluation."""
import argparse,fcntl
from .common import *
from .configuration import base
from .sustainable import duration_config,assess
from .run import execute
from .qualification import against_unseen
from .analyze import records,metrics
from .stages import state_gate

SEEDS=(2026111511,2026111512,2026111513)
SELECTION=ROOT/'configs/final_confirmation_selection.json'


def freeze():
    from .checkpoint_integrity import record
    record()
    selection=json.loads(SELECTION.read_text())
    assert selection['model_revision']==MODEL_REVISION
    assert set(selection['methods'])=={'Native','Colocated','Static','Dynamic'}
    assert json.loads((ROOT/'analysis/unseen_reference_correctness.json').read_text())['passed']
    assert json.loads((ROOT/'analysis/immutable_prompt_pool_GPU_gate.json').read_text())['passed']
    assert not list((ROOT/'raw').glob('heldout_*')),'Selection must precede held-out performance'
    hashes={str(p.relative_to(REPO)):digest(p) for p in SOURCE.glob('*.py')}
    for p in (ROOT/'configs/frozen_SLO_regimes.json',ROOT/'configs/coalesced_cost_model_v2.json',
              ROOT/'configs/discovery_residual_guard_v1.json',ROOT/'data/unseen_confirmation.json',SELECTION):
        hashes[str(p.relative_to(REPO))]=digest(p)
    for entry in selection['methods'].values():
        if entry['options'].get('cost_model'):
            p=ROOT/'configs'/entry['options']['cost_model']
            hashes[str(p.relative_to(REPO))]=digest(p)
    for relative in ('configs/verified_ByteLevel_text_priming.json',
                     'state/canonical_block_runtime_contract.json',
                     'configs/canonical_completion_residual_v1.json',
                     'state/installed_runtime_sources.json','state/frozen_predecessor_hashes.json',
                     'data/unseen_manifest.json','state/model_checkpoint_file_inventory.json'):
        p=ROOT/relative
        if p.exists():hashes[str(p.relative_to(REPO))]=digest(p)
    relative_rate=selection['methods']['Dynamic']['rate']
    value=dict(time=time.time(),hashes=hashes,selection=selection,arrival_seeds=list(SEEDS),
        measurement_seconds=600.,warm_seconds=60.,follow_seconds=90.,joint_target=.95,
        relative_check_rate=relative_rate,relative_check_methods=['Native','Dynamic'],
        design='Three independent starts per selected operating point. Reverse or rotate method order by seed. Pair Native and Dynamic at the selected Dynamic capacity rate for the relative Decode gate. The overloaded Native reference remains fully followed and is not called sustainable. Follow all measurement arrivals. Do not retune method parameters on confirmation. Persistence and upper-neighbor runs follow the declared budget rule.',
        capacity_scope='Largest confirmed tested operating point, not an exact global maximum. A single rejected upper neighbor gives a tested bracket, not an infinite-horizon proof.')
    path=ROOT/'state/heldout_runtime_freeze.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['hashes']==hashes and old['selection']==selection
    else:write(path,value)
    print('CONFIRMATION_FROZEN',digest(path),flush=True)


def expected_states():
    gate=json.loads((ROOT/'analysis/unseen_reference_correctness.json').read_text())
    return json.loads((ROOT/'raw'/gate['reference']/'full_prompt_state_prefill.json').read_text())['requests']


def qualify():
    selected=json.loads(SELECTION.read_text());expected=expected_states();results={}
    for method in ('Native','Colocated','Dynamic'):
        entry=selected['methods'][method];options=dict(entry['options'])
        options.update(dataset='unseen_confirmation.json',intern_immutable_prompt_pool=True,
            full_prompt_state_witness=True,correctness_only=True,clean_timing=False,
            phase='FROZEN_UNSEEN_STATE_AND_LIFETIME_QUALIFICATION_NOT_PERFORMANCE')
        arm=entry['arm']
        count=32
        if method=='Dynamic':
            arm='FORCED_QUALIFICATION'
            # Forced cuts advance per source macro, not per request. Five
            # macros are required to exercise every candidate at group8.
            count=max(32,5*options.get('P_group_size',4))
            options.update(request_caps=[2]*count,arrival_offsets=[0.]*count,arrival_rate=None,
                forced_layers=[0,2,4,8,12],verify_layer_execution=True,
                hidden_verify_hash=True,verify_native_prefix_KV=True,
                full_prompt_state_witness_roles=['prefill','decode'],negotiation_timeout_s=120)
        else:
            options.update(post_graph_state_witness=True,
                full_prompt_state_witness_roles=['replica0','replica1'] if method=='Colocated' else ['prefill'])
            if method=='Native':options['imported_native_state_witness']=True
        name=f'qualification_unseen_frozen_{method}_N{count}_v1'
        cfg=base(name,arm,count=count,cap=16,rate=3.2,seed=SEEDS[0],**options)
        assert execute(save(cfg),restore=False);token=against_unseen(name);assert token['passed']
        folder=ROOT/'raw'/name;states={};checks={}
        if method=='Dynamic':
            work=state_gate(name,count);assert work['passed']
            assert set(work['cuts'].values())=={24,28,32,34,36},'Every cut must actually execute'
            source=json.loads((folder/'full_prompt_state_prefill.json').read_text())['requests']
            destination=json.loads((folder/'full_prompt_state_decode.json').read_text())['requests']
            states={rid:(source[rid] if cut==36 else destination[rid]) for rid,cut in work['cuts'].items()}
            imports={r['external_request_id']:r for p in folder.glob('extension_decode_*.jsonl')
                for r in records(p) if r['event']=='NATIVE_IMPORTED_PREFIX_IDENTITY'}
            assert all(imports.get(rid,{}).get('passed',False) for rid,cut in work['cuts'].items() if cut==36)
        else:
            for role in cfg['full_prompt_state_witness_roles']:
                states.update(json.loads((folder/f'full_prompt_state_{role}.json').read_text())['requests'])
            if method=='Native':
                imported=json.loads((folder/'imported_native_prompt_KV.json').read_text())['requests']
                assert len(imported)==count
                assert all(imported[rid]['KV']==expected[rid]['KV'] for rid in states)
        for rid,value in states.items():
            checks[rid]=dict(KV_equal=value['KV']==expected[rid]['KV'],
                Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256'])
        passed=len(checks)==count and all(all(v.values()) for v in checks.values())
        results[method]=dict(name=name,passed=passed,checks=checks,output=token,evidence='CORRECTNESS_ONLY')
        write(ROOT/'analysis/frozen_unseen_full_state_qualification.json',results)
        assert passed,method+' failed unseen full-state qualification'
    print('FROZEN_UNSEEN_STATE_PASS',flush=True)


def one(label,method,rate,seed,measure=600.):
    selected=json.loads(SELECTION.read_text());entry=selected['methods'][method]
    name=f'heldout_{label}_{method}_O16_q{rate}_s{seed}_W{int(measure)}_v1'
    options=dict(entry['options']);options.update(duration_prompt_pool='unseen_confirmation.json',
        intern_immutable_prompt_pool=True,phase='FROZEN_INDEPENDENT_CONFIRMATION',
        selected_method_label=method,selection_SHA256=digest(SELECTION))
    cfg=duration_config(name,entry['arm'],rate,cap=16,seed=seed,warm=60.,measure=measure,follow=90.,**options)
    assert execute(save(cfg),restore=False);assert against_unseen(name)['passed']
    value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    append(ROOT/'state/independent_confirmation_progress.jsonl',dict(name=name,time=time.time(),
        stage=label,method=method,rate=rate,seed=seed,
        qualified={k:v['qualified'] for k,v in value['SLO_regimes'].items()}))
    print('HELDOUT',name,value['raw_completed_output_tokens_s'],
        {k:(v['qualified'],v['arrival_cohort_joint_attainment']) for k,v in value['SLO_regimes'].items()},flush=True)
    return name


def run():
    selected=json.loads(SELECTION.read_text());assert (ROOT/'state/heldout_runtime_freeze.json').exists()
    state=json.loads((ROOT/'analysis/frozen_unseen_full_state_qualification.json').read_text())
    assert set(state)=={'Native','Colocated','Dynamic'} and all(v['passed'] for v in state.values())
    orders=[('Native','Colocated','Static','Dynamic'),('Dynamic','Static','Colocated','Native'),
        ('Colocated','Dynamic','Native','Static')]
    names=[]
    for seed,order in zip(SEEDS,orders):
        for method in order:names.append(one('capacity',method,selected['methods'][method]['rate'],seed))
        # Compare relative Decode tails at the actual Dynamic operating rate.
        # A low-load comparison can hide the batch geometry at capacity.
        # An overloaded Native reference is followed, not called sustainable.
        native=selected['methods']['Native']
        relative_rate=selected['methods']['Dynamic']['rate']
        if native['rate']!=relative_rate:names.append(one('relative','Native',relative_rate,seed))
    skipped=[]
    stop_new=selected['stop_new_GPU_epoch']
    for method in ('Colocated','Dynamic'):
        rate=selected['methods'][method]['rate']
        if time.time()+1500>stop_new:
            skipped.append(dict(stage='persistence',method=method,status='NOT_RUN_BUDGET'))
            continue
        names.append(one('persistence',method,rate,2026111515,measure=1200.))
    for method in ('Native','Colocated','Static','Dynamic'):
        rate=selected['methods'][method].get('upper_rate')
        if rate is None:continue
        if time.time()+1000>stop_new:
            skipped.append(dict(stage='upper_neighbor',method=method,status='NOT_RUN_BUDGET'))
            continue
        names.append(one('upper_neighbor',method,rate,2026111514))
    write(ROOT/'analysis/independent_confirmation_complete.json',dict(time=time.time(),names=names,
        skipped=skipped,evidence='MEASURED',scope='All executed starts and operating points remain in the final analysis, including rejected points. Optional persistence and upper neighbors use the frozen time-budget rule. Skipped points are NOT_RUN, not failures. No method retuning uses these outcomes.'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','qualify','run'])
    stage=parser.parse_args().stage
    if stage=='freeze':freeze();return
    try:(qualify if stage=='qualify' else run)()
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
