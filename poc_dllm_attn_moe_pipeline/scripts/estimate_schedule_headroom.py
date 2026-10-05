"""Export dependency-safe natural bound and explicitly limited split estimates."""
import csv
import json
from collections import defaultdict
import numpy as np
from poc_dllm_attn_moe_pipeline.contract import *


def bootstrap(values, seed=10062026):
    values=np.asarray(values,float)
    if len(values)==0:
        return None
    rng=np.random.default_rng(seed)
    draws=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),n_requests=len(values))


def csv_write(path, rows, fields=None):
    if fields is None:
        fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    out=ROOT/'results/main'
    raw=ROOT/'raw/main_v1'
    freeze=json.loads((out/'design_freeze.json').read_text())
    selected={r['request_id']:r for r in freeze['rows'] if r['early_kill']}
    requests=[]; readiness=[]; nodes=[]; phases=[]; observer=[]
    parity=True
    mask_audit=[]
    for rid,row in selected.items():
        clean=[json.loads((raw/f'{rid}.clean.{rep}.rank0.json').read_text()) for rep in (0,1)]
        event_rows=[json.loads((raw/f'{rid}.events.0.rank{rank}.json').read_text()) for rank in range(4)]
        trace=[json.loads((raw/f'{rid}.trace.0.rank{rank}.json').read_text()) for rank in range(4)]
        # One cumulative-critical rank per request; no max-of-components sum.
        critical=max(range(4),key=lambda r:sum(c['end_us']-c['start_us'] for c in event_rows[r]['calls']))
        sample=event_rows[critical]
        check=all(x['output_ids']==clean[0]['output_ids'] and x['nfe']==clean[0]['nfe']
                  for x in clean+event_rows+trace)
        acceptance_check=all(x['acceptances']==event_rows[0]['acceptances'] for x in event_rows+trace)
        parity &= check and acceptance_check
        clean_ms=float(np.mean([x['wall_ms'] for x in clean]))
        overhead=100*(sample['wall_ms']/clean_ms-1)
        if row['split']=='heldout':
            observer.append(overhead)
        stages=defaultdict(float)
        for e in sample['events']:
            stages[e['stage']]+=e['duration_us']/1000
        requests.append(dict(request_id=rid,split=row['split'],variant='B0_PINNED_NATIVE',
            prompt_tokens=clean[0]['prompt_tokens'],returned_tokens=len(clean[0]['output_ids']),
            truncated='BOUNDED64_NOT_TASK_SCORE',nfe=clean[0]['nfe'],e2e_clean_ms=clean_ms,
            ep_dispatch_ms=stages['dispatch'],ep_combine_ms=stages['combine'],
            tp_layout_ms=stages['source_all_gather'],attention_ms=stages['attention'],
            observer_overhead_pct=overhead,token_parity=check,acceptance_parity=acceptance_check,
            task_score='NOT_RUN',reference_variant='native self-repeat',critical_rank=critical,
            detailed_attribution_valid=abs(overhead)<=3))
        # Native DAG: externally visible operators are serialized. Include gaps
        # once, not the inclusive nested router/dispatch/expert components twice.
        phase_names={'prepare_attn','attention','prepare_mlp','moe','postprocess_layer'}
        ordered=sorted([e for e in sample['events'] if e['stage'] in phase_names],key=lambda e:e['start_us'])
        prior_node=None;cursor=0.;duration=0.
        for index,e in enumerate(ordered):
            gap=max(0.,e['start_us']-cursor)
            if gap:
                name=f'{rid}:gap{index}'
                nodes.append(dict(node_id=name,request_id=rid,block_id=None,refinement_id=None,model_invocation_id=e['call_id'],
                    layer_id=e['layer'],rank=critical,group_id='full',operator='host_or_unattributed_gap',
                    input_version='',output_version='',predecessors=json.dumps([] if prior_node is None else [prior_node]),
                    measured_duration_us=gap,resource_class='native_stream_timeline',
                    visibility_requirement='ordered completion',cost_source='same-device events; NOT validated headline'))
                prior_node=name;duration+=gap
            name=f'{rid}:op{index}'
            nodes.append(dict(node_id=name,request_id=rid,block_id=None,refinement_id=None,model_invocation_id=e['call_id'],
                layer_id=e['layer'],rank=critical,group_id='full',operator=e['stage'],
                input_version=f'model-call{e["call_id"]}:layer{e["layer"]};KV-version-not-traced',output_version='native fresh',
                predecessors=json.dumps([] if prior_node is None else [prior_node]),
                measured_duration_us=e['duration_us'],resource_class='native GPU current stream',
                visibility_requirement='whole-buffer release; all TP heads',cost_source='direct same-device CUDA events'))
            prior_node=name;duration+=e['duration_us'];cursor=e['end_us']
        phases.append(dict(request_id=rid,rank=critical,reconstructed_timeline_us=duration,
                           measured_last_stage_offset_us=cursor,reconstruction_error_pct=100*(duration/cursor-1),
                           native_dependency_envelope_pct=0.,engineered_whole_request_envelope_pct=None,
                           unknown_reason='No task-group publication primitive; split estimates not whole-request timing'))
        for rank,t in enumerate(event_rows):
            by_call={c['call_id']:c for c in t['calls']}
            for e in t['events']:
                if e['stage']!='source_all_gather':
                    continue
                c=by_call[e['call_id']]
                for group in range(2):
                    readiness.append(dict(request_id=rid,problem_id=row['problem_id'],block_id=None,
                        refinement_id=None,model_invocation_id=e['call_id'],layer_id=e['layer'],source_rank=rank,group_id=group,
                        group_size=c['rows']//8,phase=c['phase'],prefix_tokens=c['positions'][0][0] if c['cache_present'] else 0,
                        current_block_tokens=c['rows'],completion_scope='WHOLE_BUFFER',natural_or_engineered='natural',
                        r_math_us=None,r_publish_us=e['start_us'],r_tp_visible_us=e['end_us'],r_all_us=e['end_us'],
                        clock_domain=t['clock_domain'],clock_error_us=None,
                        release_primitive='native combine wait + shared/add + NCCL all_gather completion',
                        trace_mode='events',unknown_reason='latent row math/cross-GPU alignment unobservable; native iteration/block not directly joined to call; same-device release only'))
    components=[json.loads((ROOT/f'raw/components_v3/components.rank{rank}.json').read_text()) for rank in range(4)]
    producer=[]; consumers=[]; split_ratios=[]; windows=[]; numeric=[]; shared=[]
    for rank,payload in enumerate(components):
        for e in payload['records']:
            if e['kind']=='producer':
                producer.append(dict(rank=rank,**e))
                numeric.append(e['max_abs'] <= .03125 and e['relative_l2'] <= .002)
                if e['tp_visible_gap_us'] is not None:
                    windows.append(e['tp_visible_gap_us'])
                    for g in range(2):
                        readiness.append(dict(request_id='coding_00',problem_id=selected['coding_00']['problem_id'],
                            block_id=None,refinement_id=None,model_invocation_id=None,layer_id=e['layer'],source_rank=rank,group_id=g,
                            group_size=4,phase='hot-calibration-component',prefix_tokens=64,current_block_tokens=32,
                            completion_scope='ENGINEERED_GROUP_MOE_ROWS',natural_or_engineered='engineered',
                            r_math_us=None,r_publish_us=None,r_tp_visible_us=-e['tp_visible_gap_us'] if g==0 else 0.,
                            r_all_us=0.,clock_domain=f'rank{rank}:producer-repeat{e["repetition"]}:last-group-release=0',
                            clock_error_us=None,release_primitive='group DeepEP wait/shared/add/all-gather/scatter completion event',
                            trace_mode='physical-component',unknown_reason='No latent/publication subevents or global clocks; not a full two-layer seam'))
            elif e['kind']=='consumer':
                consumers.append(dict(rank=rank,**e))
                mask_audit.append(dict(rank=rank,layer=e['layer'],native_mask_none=e['native_attention_mask_none'],
                    logical_prefix_tokens=e['logical_prior_prefix_tokens'],physical_prefix_slots=e['prefix_cache_slots'],
                    padded_slots=e['padded_prefix_slots'],current_fresh=True,
                    note='Preserved native unmasked padded-cache contract; padding slots are not extra real prompt tokens'))
            elif e['kind']=='generic_shared':
                shared.append(dict(rank=rank,**e))
    producer_aggregate=[]
    for lid in (1,9,17):
        for rep in range(5):
            pair=[e for e in producer if e['layer']==lid and e['repetition']==rep]
            # Critical rank is selected by combined pair cost, fixed for both arms.
            rank=max(range(4),key=lambda r:sum(e['duration_us'] for e in pair if e['rank']==r))
            u=next(e['duration_us'] for e in pair if e['rank']==rank and e['variant']=='unsplit')
            s=next(e['duration_us'] for e in pair if e['rank']==rank and e['variant']=='split_serial')
            split_ratios.append(s/u)
            producer_aggregate.append(dict(layer=lid,repetition=rep,critical_rank=rank,unsplit_us=u,
                split_us=s,split_ratio=s/u,split_tax_us=s-u))
    csv_write(out/'readiness.csv',readiness)
    csv_write(out/'dependency_nodes.csv',nodes)
    csv_write(out/'request_metrics.csv',requests)
    csv_write(out/'dependency_reconstruction.csv',phases)
    csv_write(out/'producer_split_metrics.csv',producer)
    csv_write(out/'producer_paired.csv',producer_aggregate)
    csv_write(out/'seam_metrics.csv',[],['request_id','boundary','variant','group_plan','prefix_tokens','repetition','order',
        'producer_us','layout_us','qkv_us','prefix_attention_us','current_attention_us','merge_us','shared_us',
        'final_collective_us','control_us','seam_makespan_us','extra_hbm_bytes','correctness_status','clock_quality'])
    write(out/'consumer_components.json',consumers)
    write(out/'generic_shared_components.json',shared)
    write(out/'cache_mask_audit.json',mask_audit)
    write(out/'bootstrap_ci.json',dict(unit='8 untouched heldout requests; repeated measurements aggregated per request',
        observer_overhead_pct=bootstrap(observer),natural_ready_window_us=bootstrap([0.]*8),
        engineered='one calibration snapshot; repetition/component evidence only, no request-generalization CI'))
    headroom=dict(native_published_dependency_only_e2e_envelope_pct=0.,
        native_unobservable_latent_headroom=None,engineered_dependency_only_whole_request_envelope=None,
        engineered_measured_cost_whole_request_estimate=None,
        reason='Natural whole-buffer release has no early TP-visible group. Engineered split creates local windows but pays >2x producer tax; global clock/strong baseline attribution inadequate for an honest whole-request projection.',
        natural_ready_window_us=0.,natural_tp_visible_window_us=0.,
        engineered_same_device_tp_visible_window_us=dict(median=float(np.median(windows)),min=min(windows),max=max(windows)),
        producer_split_ratio_median=float(np.median(split_ratios)),
        consumer_work='native norm/QKV/QK norm/RoPE and native-valid immutable cache only; fresh bidirectional current K/V join remains mandatory',
        shared_residual_layout_preserved=True,simulated_speedup=False)
    write(out/'headroom.json',headroom)
    attention_failures=[dict(rank=e['rank'],layer=e['layer'],max_abs=e['max_abs'],relative_l2=e['relative_l2'])
        for e in consumers if e['max_abs'] > .03125 or e['relative_l2'] > .003]
    write(out/'correctness.json',dict(native_tokens_nfe_parity=parity,acceptance_trace_vs_event_parity=parity,
        measured_native_requests=12,producer_split_output_parity=all(numeric),
        independent_fp32_merge_max_abs=max(e['fp32_reference_max_abs'] for e in consumers),
        frozen_bf16_reference_failures=attention_failures,tolerances_never_relaxed=True,
        pipeline_candidate='NOT_IMPLEMENTED',full_runtime_discrete_parity='NOT_RUN',
        cache_current_kv='fresh; native unmasked padding retained, no accepted-current immutable-prefix claim'))
    write(out/'accuracy.json',dict(GSM8K='NOT_RUN',HumanEval='NOT_RUN',reason='64-token bounded prefixes, no candidate runtime; generated code was never executed'))
    summary=dict(verdict='CONDITIONAL_INTERFACE_BLOCKED',stage_b='NOT_RUN',stage_c='NOT_RUN',
        actual_epoch='EPOCH_RUNTIME_NOT_RUN',prior_overlap='Substantial generic overlap; no novelty/superiority claim',
        headroom=headroom,native_parity=parity,producer_split_parity=all(numeric),
        observer_mean_overhead_pct=float(np.mean(observer)),observer_attribution='UNVALIDATED above3% target; clean E2E preserved',
        candidate_seam_gain_vs_unsplit=None,candidate_seam_gain_vs_fixed=None,measured_full_e2e_gain=None,
        stop_reason='Stock whole-buffer EP+source-all-gather publication; engineered2-group tax; no validated fine-grained visibility/whole-request gain. Late calibration BF16 partial attention reference exceeds frozen absolute tolerance.',
        next_action='Do not build a scheduler. Smallest remaining test: supported safe group publication+layout primitive under a graph-compatible generic fixed pipeline, before any prefix-specific gain claim.',
        gpu_mask=MASK,burner_final='InitialOFF restored per authoritative §6; no new service')
    write(out/'gate_audit.json',dict(G0=dict(physical_EP4=True,ownership=True,native_parity=parity,
        cross_device_clock_alignment='UNRESOLVED',observer='UNVALIDATED',candidate_correctness='NOT_RUN'),
        G1=dict(natural_interface='ZERO_EARLY_VISIBLE_HEADROOM',engineered='COST_AND_VISIBILITY_BLOCKED',latent='UNKNOWN'),
        G2='NOT_RUN',G3='NOT_RUN',G4='NO_PROMOTION'))
    write(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
