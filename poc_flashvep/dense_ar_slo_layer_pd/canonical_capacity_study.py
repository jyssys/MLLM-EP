"""Test generic layout changes in both PD and the strong Colocated path."""
import fcntl
import math
from .common import *
from .configuration import base
from .generic_qualification import independent
from .objective_rate_discovery import graph_options
from .expired_deadline_recovery import options
from .run import execute
from .qualification import against_previous
from .sustainable import duration_config,assess
from .analyze import metrics


def wait_for_mechanism():
    end=json.loads((ROOT/'state/progress.json').read_text())['start_epoch']+15.*3600
    while time.time()<end:
        path=ROOT/'analysis/canonical_transport_ABA.json'
        if path.exists():
            value=json.loads(path.read_text())
            if not value['continue_calibration']:return False
            if (ROOT/'state/canonical_transport_study_complete.json').exists():return True
        time.sleep(10)
    raise TimeoutError('Preserve the study if its mechanism phase does not finish before the discovery cutoff')


def colocated_qualification():
    name='qual_canonical_C_FULL512_N32_full_state_v1'
    changes=graph_options(512)
    changes.update(canonical_new_KV_block_order=True,intern_immutable_prompt_pool=True,
        full_prompt_state_witness=True,full_prompt_state_witness_roles=['replica0','replica1'],
        post_graph_state_witness=True,correctness_only=True,clean_timing=False,
        phase='GENERIC_LAYOUT_STRONG_COLOCATED_FULL_STATE_QUALIFICATION')
    cfg=base(name,'Colocated',count=32,cap=16,rate=5.,**changes)
    assert execute(save(cfg),restore=False);output=against_previous(name);assert output['passed']
    expected=independent();actual={}
    for role in ('replica0','replica1'):
        actual.update(json.loads((ROOT/'raw'/name/f'full_prompt_state_{role}.json').read_text())['requests'])
    checks={rid:dict(KV_equal=value['KV']==expected[rid]['KV'],
        Hidden_equal=value['final_hidden_SHA256']==expected[rid]['final_hidden_SHA256']) for rid,value in actual.items()}
    passed=len(checks)==32 and all(all(value.values()) for value in checks.values())
    write(ROOT/'analysis/canonical_C_full_state_gate.json',dict(name=name,passed=passed,checks=checks,
        output=output,evidence='CORRECTNESS_ONLY'))
    assert passed


def colocated_ABA():
    values=[]
    for tag,enabled in (('A1',False),('B',True),('A2',False)):
        name=f'canonical_C_FULL512_{tag}_N128_O16_q5.0_v1'
        cfg=base(name,'Colocated',count=128,cap=16,rate=5.,
            canonical_new_KV_block_order=enabled,intern_immutable_prompt_pool=True,
            phase='COLOCATED_LAYOUT_TUNING_NOT_CONFIRMATION',**graph_options(512))
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        value=metrics(name);write(ROOT/'analysis'/(name+'_metrics.json'),value);values.append(value)
    def score(value):return value['SLO_regimes']['Decode_priority']['Output_token_SLO_goodput']
    keep=score(values[1])>=(score(values[0])+score(values[2]))/2
    write(ROOT/'analysis/canonical_C_layout_ABA.json',dict(runs=[v['name'] for v in values],
        Decode_priority_scores=[score(v) for v in values],selected_canonical_layout=keep,
        evidence='MEASURED',scope='One finite-cohort A/B/A tuning sequence. Only layout ordering changes. This is not an independent confidence interval or sustainable throughput. Select by the same unchanged Decode-priority objective.'))
    return keep


def completion_margin():
    from .prediction_validation import analyze
    values=[]
    for seed in (2026111464,2026111465,2026111466):
        name=f'canonical_completion_cal_DIRECT_N128_q5.0_s{seed}_v1'
        changes=options(True)
        changes.update(canonical_new_KV_block_order=True,intern_immutable_prompt_pool=True,
            cost_model='canonical_cost_model_v5.json',phase='POST_LAYOUT_COMPLETION_RESIDUAL_CALIBRATION')
        cfg=base(name,'DIRECT',count=128,cap=16,rate=5.,seed=seed,**changes)
        assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
        value=analyze(name)
        assert value['errors']['group_first_output_nominal']['samples']>=8,'Insufficient actual migrated groups'
        write(ROOT/'analysis'/(name+'_prediction.json'),value)
        write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name));values.append(value)
    largest=max(v['errors']['group_first_output_nominal']['maximum_underprediction_ms'] for v in values)
    margin=max(110.,10*math.ceil((largest+20.)/10))
    result=dict(time=time.time(),names=[v['name'] for v in values],independent_starts=3,
        maximum_nominal_group_underprediction_ms=largest,total_margin_ms=margin,
        rule='Maximum observed positive nominal group-completion residual plus20ms. Round upward to10ms. Keep a110ms floor. Do not delete delivery outliers or select the margin by later serving goodput.',
        evidence='TRACE_ANALYSIS',scope='Changed transport and async runtime. Use independent calibration starts, not held-out confirmation. Groups within each start remain dependent. This empirical envelope is not WCET or a future p99 guarantee. Actual cohort-retirement guards remain unchanged.')
    write(ROOT/'configs/canonical_completion_residual_v1.json',result)
    print('CANONICAL_COMPLETION_MARGIN',margin,flush=True)
    return margin


def long_one(label,arm,rate,changes):
    name=f'canonical_capacity_{label}_O16_q{rate}_s2026111462_W600_v1'
    changes=dict(changes,canonical_new_KV_block_order=True,intern_immutable_prompt_pool=True,
        cost_model='canonical_cost_model_v5.json',phase='GENERIC_LAYOUT_CAPACITY_DISCOVERY')
    cfg=duration_config(name,arm,rate,seed=2026111462,warm=60.,measure=600.,follow=90.,**changes)
    assert execute(save(cfg),restore=False);assert against_previous(name)['passed']
    value=assess(name);write(ROOT/'analysis'/(name+'_metrics.json'),metrics(name))
    print('CANONICAL_CAPACITY',name,value['raw_completed_output_tokens_s'],
        {k:v['qualified'] for k,v in value['SLO_regimes'].items()},flush=True)
    return name


def main():
    try:
        if not wait_for_mechanism():
            write(ROOT/'state/canonical_capacity_study_complete.json',dict(time=time.time(),
                status='NOT_RUN_MECHANISM_GATE',scope='No new capacity claim without a positive mechanism test.'))
            return
        from .post_layout_decode_calibration import build
        build()
        colocated_qualification();keep=colocated_ABA();margin=completion_margin();names=[]
        native=dict(verified_ByteLevel_priming=True,direct_phase_norm=True,async_decode=True,
            nixl_manual_coalesce=True,nixl_descriptor_sort=False)
        names.append(long_one('Native','Native',3.4,native))
        static=options(True);static.update(static_layers=12,migration_fraction=1.)
        names.append(long_one('Static12','STATIC',5.,static))
        dynamic=options(True);dynamic.update(safety_margin_ms=margin,
            safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)})
        names.append(long_one('Dynamic','DIRECT',5.,dynamic))
        # Smaller destination cohorts protect the Native-relative Decode
        # tail. Charge the extra intercohort residence in the same frontier.
        stream=options(False);stream.update(D_admission_batch=4,safety_margin_ms=margin,
            safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)})
        names.append(long_one('Static12_D4','STATIC',4.8,dict(stream,static_layers=12,migration_fraction=1.)))
        names.append(long_one('Dynamic_D4','DIRECT',4.8,stream))
        write(ROOT/'state/canonical_capacity_study_complete.json',dict(time=time.time(),
            status='COMPLETE',names=names,Colocated_layout_selected=keep,completion_margin_ms=margin,
            scope='Generic layout is shared by all PD arms. Colocated is tuned independently for its own SLO objective. No layer-policy contribution follows from a layout improvement alone.'))
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
