"""Freeze predeclared discovery choices and complete the independent campaign."""
import fcntl
from .common import *
from .objective_rate_discovery import graph_options
from .expired_deadline_recovery import options
from .confirmation import freeze,qualify,run,SELECTION


def wait_discovery():
    cutoff=json.loads((ROOT/'state/progress.json').read_text())['start_epoch']+16*3600
    while time.time()<cutoff:
        path=ROOT/'state/canonical_capacity_study_complete.json'
        if path.exists() and (ROOT/'state/final_Static_fraction_discovery_complete.json').exists():
            value=json.loads(path.read_text());assert value['status']=='COMPLETE',value
            return value
        time.sleep(10)
    raise TimeoutError('Discovery did not finish before the predeclared confirmation cutoff')


def point(label,rate):
    name=f'canonical_capacity_{label}_O16_q{rate}_s2026111462_W600_v1'
    value=json.loads((ROOT/'analysis'/(name+'_sustainable_v2.json')).read_text())
    return name,value['SLO_regimes']['Decode_priority']['qualified']


def assert_runtime_matches(name):
    folder=ROOT/'raw'/name;hashes=json.loads((folder/'source_hashes.json').read_text())
    critical=('worker.py','native.py','dynamic_native.py','policy.py','deadline.py','credits.py',
        'canonical_blocks.py','dynamic_extension.py','extension.py','hidden_transport.py',
        'exact_phase_norm.py','phase_mask_norm.py','phase_mask_runtime.py','output_priming.py',
        'fast_text_priming.py','input_pool.py','manual_coalescing.py','async_ingress.py')
    for name in critical:
        relative=Path('poc_flashvep/dense_ar_slo_layer_pd')/name
        expected=hashes[str(folder/'worker_source'/relative)]
        assert digest(REPO/relative)==expected,'Selected execution source changed: '+str(relative)


def selection(discovery):
    margin=json.loads((ROOT/'configs/canonical_completion_residual_v1.json').read_text())['total_margin_ms']
    choices={};source_points={}
    for method,whole_label,small_label in (('Static','Static12','Static12_D4'),('Dynamic','Dynamic','Dynamic_D4')):
        whole,whole_ok=point(whole_label,5.);small,small_ok=point(small_label,4.8)
        if whole_ok:wait=True;rate=5.;upper=5.2;chosen=whole;status='QUALIFIED_DISCOVERY_POINT'
        elif small_ok:wait=False;rate=4.8;upper=5.;chosen=small;status='QUALIFIED_DISCOVERY_POINT'
        else:
            wait=False;rate=4.4;upper=4.8;chosen=small;status='PREDECLARED_UNTESTED_LOWER_RATE_CANDIDATE'
        assert_runtime_matches(chosen)
        cfg=options(wait);cfg.update(canonical_new_KV_block_order=True,
            cost_model='canonical_cost_model_v5.json',safety_margin_ms=margin,
            safety_margin_by_layers_ms={str(k):margin for k in (2,4,8,12)})
        if not wait:cfg['D_admission_batch']=4
        if method=='Static':cfg.update(static_layers=12,migration_fraction=1.)
        if method=='Static':
            tuned=json.loads((ROOT/'analysis/final_Static_fraction_discovery.json').read_text())['long_selected']
            if tuned and tuned['qualified'] and rate<=4.8:
                rate=4.8;upper=5.;chosen=tuned['name'];status='QUALIFIED_TUNED_STATIC_FRACTION'
                cfg.update(D_wait_for_source_macro=False,D_admission_batch=4,
                    migration_fraction=tuned['fraction'],static_request_block_size=8)
                assert_runtime_matches(chosen)
        choices[method]=dict(arm='STATIC' if method=='Static' else 'DIRECT',rate=rate,
            upper_rate=upper,options=cfg,selection_status=status)
        source_points[method]=dict(chosen_geometry=chosen,whole_point=whole,whole_qualified=whole_ok,
            D4_point=small,D4_qualified=small_ok)
    # The new3.4 point is an actual rejected upper neighbor. Use the lower3.2
    # candidate for confirmation, not a failed point labeled as capacity.
    native_name,native_ok=point('Native',3.4)
    assert not native_ok,'Review an unexpected changed Native frontier before choosing its operating point'
    assert_runtime_matches(native_name)
    choices['Native']=dict(arm='Native',rate=3.2,upper_rate=3.4,
        selection_status='LOWER_RATE_AFTER_REJECTED_CANONICAL_3.4_POINT',options=dict(
            verified_ByteLevel_priming=True,direct_phase_norm=True,async_decode=True,
            nixl_manual_coalesce=True,nixl_descriptor_sort=False,
            canonical_new_KV_block_order=True,cost_model='canonical_cost_model_v5.json'))
    layout=json.loads((ROOT/'analysis/canonical_C_layout_ABA.json').read_text())
    C=graph_options(512);C['canonical_new_KV_block_order']=layout['selected_canonical_layout']
    choices['Colocated']=dict(arm='Colocated',rate=5.,upper_rate=5.2,options=C,
        selection_status='SLO_SPECIFIC_FULL_GRAPH_FRONTIER_AFTER_ROUTER_AND_LAYOUT_TUNING')
    start=json.loads((ROOT/'state/progress.json').read_text())['start_epoch']
    value=dict(time=time.time(),model_revision=MODEL_REVISION,methods=choices,
        stop_new_GPU_epoch=start+20*3600,source_points=source_points,
        calibration=dict(model='canonical_cost_model_v5.json',model_SHA256=digest(ROOT/'configs/canonical_cost_model_v5.json'),
            completion_margin_ms=margin,margin_SHA256=digest(ROOT/'configs/canonical_completion_residual_v1.json')),
        selection_rule='Choose the highest passed predeclared Decode-priority discovery point for each PD method. Whole D8 at5.0 precedes streaming D4 at4.8. If both fail, test the predeclared lower4.4 D4 candidate. It is not called qualified before confirmation. Native uses3.2 after the rejected new3.4 point. Colocated uses its stronger SLO-specific5.0 frontier. Do not retune on unseen performance.',
        scope='Capacity candidates are selected before any held-out performance. An untested lower point remains a candidate. Same-rate Native references follow the selected Dynamic rate. No exact global maximum or untested workload benefit is assumed.')
    assert not SELECTION.exists(),'Do not replace an existing frozen selection'
    write(SELECTION,value);return value


def main():
    discovery=wait_discovery();selection(discovery);freeze()
    try:qualify();run()
    finally:
        with QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not resources().processes():resources().restore(list(range(8)))


if __name__=='__main__':main()
