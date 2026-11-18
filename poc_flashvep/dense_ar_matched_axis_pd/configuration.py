"""Freeze equal caps, exact row plans and fixed partition sizes."""
import json
from .common import PREVIOUS, ROOT
from ..duetserve_agg_decode.configuration import offsets, standard_capture_sizes
from ..dense_ar_layer_token_pd.token_axis import matched_token_plan


def destination_limit(indices, cfg, maximum):
    """Apply one query-budget admission rule to every PD method."""
    used, count = 0, 0
    for index in indices[:maximum]:
        if cfg.get('token_axis'):
            rows = cfg.get('token_suffix_by_request', [cfg['token_suffix']]*cfg['count'])[index]
        elif cfg.get('policy') == 'STATIC':
            rows = cfg['length']
        else:
            rows = 1
        if used + rows > cfg['D_token_budget']:
            break
        used += rows
        count += 1
    assert count > 0
    return count


def atomic_base(name, method='Native', layers=12, count=144, cap=16, rate=6., seed=2026111780,
         dataset='discovery.json', correctness=False, **changes):
    template = PREVIOUS/'configs/heldout_Static_N128_O16_q6.4_s2026111740_v1.json'
    cfg = json.loads(template.read_text())
    cfg.update(name=name, count=count, cap=cap, request_caps=[cap]*count, length=8000,
        dataset=dataset, arrival_offsets=offsets(count, rate, seed), arrival_rate=rate, arrival_seed=seed,
        arm='DP2' if method=='Agg' else 'PD_G1', evaluation_arm=method,
        phase='UNIFIED_MATCHED_AXIS_DISCOVERY', P_slots=1024, D_slots=1024,
        token_budget=8192, D_token_budget=8192, chunk=8000, atomic_prompt=True,
        P_group_size=1, D_admission_batch=8, D_cohort_collect_ms=2,
        D_wait_for_source_macro=False, D_combine_ready_macros=False,
        preplanned_source_groups=False, cost_model=None, flow_cost_model=None,
        flow_balance_policy=False, track_active_preparation=False,
        max_reserved_requests=1024, policy='STATIC' if method=='Layer' else 'NATIVE',
        static_layers=layers, migration_fraction=1., forced_layers=[],
        token_axis=method=='Token', layer_mode='MIGRATE',
        layer_cut=36-layers if layers in (2,4,8,12) else 28,
        dynamic_layer_extension=True, hidden_transport='NIXL_GPU',
        hidden_source_slots=32, hidden_destination_slots=32,
        reuse_prefill_token=method=='Native', native_continuous_D_ingress=False,
        cohort_barrier=True, batch_commands=True, ar_compile_mode=0,
        ar_cudagraph_mode='FULL_DECODE_ONLY', decode_only_graph_roles=['decode'],
        ar_async_scheduling=False, async_decode=True,
        explicit_cudagraph_capture_sizes=[1,2,4,8,16,32],
        warmup_count=1, warmup_layer_counts=[layers if method=='Layer' else 0],
        warmup_batch_sizes=[1], warmup_output_cap=16,
        full_prompt_state_witness=False, axis_state_witness=False,
        imported_native_state_witness=False, verify_native_prefix_KV=False,
        verify_layer_execution=correctness, hidden_verify_hash=correctness,
        state_capture=False, runtime_observation=correctness, minimal_geometry_counter=True,
        physical_activity=False, execution_events=False, boundary_events=False,
        clean_timing=not correctness, correctness_only=correctness,
        normal_lifetime_audit=True, full_prompt_state_witness_roles=['prefill','decode'],
        entry_module='poc_flashvep.dense_ar_matched_axis_pd.serving',
        unified_admission_policy=True, controller_credit_fast_lane=False,
        timeout=1200, wall_timeout=1650, reference_name=None)
    if method=='Token':
        cfg.update(token_suffix={4:896,8:1776,12:2672}[layers], reuse_prefill_token=False)
        if count % 9 == 0:
            cfg['token_suffix_by_request'] = matched_token_plan(layers, count, group_size=1)
        else:
            cfg.pop('token_suffix_by_request', None)
        cfg.update(layer_mode=None, hidden_transport=None)
    else:
        cfg.pop('token_suffix_by_request', None)
        cfg.pop('token_suffix', None)
    if method=='Agg':
        cfg.update(dynamic_layer_extension=False, layer_mode=None, hidden_transport=None,
            worker_state_feedback=False,
            cohort_barrier=False, ar_cudagraph_mode='FULL', decode_only_graph_roles=[],
            ar_async_scheduling=True, atomic_prompt=False, chunk=8192,
            gpu_phase_mask_norm=True, direct_phase_norm=False,
            explicit_cudagraph_capture_sizes=standard_capture_sizes(1024))
        if correctness:
            cfg['independent_model_layer_witness']=True
    cfg.update(changes)
    assert cfg['P_slots']==cfg['D_slots']==1024
    assert cfg['token_budget']==cfg['D_token_budget']==8192
    pool = json.loads((ROOT/'data'/cfg['dataset']).read_text())
    assert len(pool)>=count, 'Prepare an explicit prompt pool before launch'
    return cfg


def base(*args,**kwargs):
    cfg=atomic_base(*args,**kwargs)
    if cfg['arm']=='PD_G1':
        cfg.update(source_chunk_packing=True,P_group_size=8,atomic_prompt=False,
            source_outstanding_limit=32,verify_native_prefix_KV=False,
            imported_native_state_witness=bool(cfg.get('correctness_only') and cfg['evaluation_arm']=='Native'))
    return cfg
