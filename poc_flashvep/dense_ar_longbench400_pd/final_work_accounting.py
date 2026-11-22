"""Require terminal evidence and an explicit scientific review before closeout."""
import json
import time
from .common import ROOT,REPO,SOURCE,write,digest,preserve
from .publication import DOCUMENTS


def read(relative):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else None


def terminal_jobs(names,stopped=()):
    stopped={r['name']:r for r in stopped}
    results=[]
    for name in names:
        receipt=read('raw/'+name+'/dispatcher_result.json')
        if receipt is not None:
            results.append(dict(name=name,status='COMPLETE' if receipt['complete'] else 'FAILED',
                terminal=True,receipt='raw/'+name+'/dispatcher_result.json'))
        elif name in stopped and stopped[name].get('status')=='NOT_RUN' and stopped[name].get('reason'):
            results.append(dict(name=name,status='NOT_RUN',terminal=True,reason=stopped[name]['reason']))
        else:
            results.append(dict(name=name,status='PENDING',terminal=False))
    return dict(expected=len(names),terminal=sum(r['terminal'] for r in results),results=results,
        passed=all(r['terminal'] for r in results))


def main():
    preserve()
    checks=[]
    def add(name,passed,evidence,scope):
        checks.append(dict(name=name,passed=bool(passed),evidence=evidence,scope=scope))

    final=read('configs/final_continuous_execution_plan_v1.json')
    completed=[]
    for lane in range(4):
        value=read(f'state/final_performance_lane{lane}_complete_v1.json')
        if value:completed.extend(value.get('results',[]))
        add(f'final_lane{lane}_drained',value is not None,
            f'state/final_performance_lane{lane}_complete_v1.json','Operational completion is not a scientific GO.')
    if final:
        names=[n for cell in final['cells'] for n in cell['jobs'].values()]
        ledger=terminal_jobs(names,completed)
        add('final_static_starts_terminal',ledger['passed'],ledger,'Every frozen arm remains. A failed start is not valid latency.')
    else:add('final_static_plan_exists',False,None,'Do not close a campaign without its final plan.')

    bridge=read('configs/controlled_continuous_I3_bridge_frozen_v1.json')
    completion=read('state/controlled_continuous_bridge_complete_v1.json')
    if bridge and completion:
        ledger=terminal_jobs(bridge['performance_jobs'],completion['results'])
        add('controlled_C3_bridge_terminal',ledger['passed'],ledger,
            'The bridge keeps strong Stock Token and the diagnostic Mode0 control separate.')
    else:add('controlled_C3_bridge_terminal',False,None,'All three declared bridge conditions require terminal outcomes.')

    repair=read('configs/hybrid_READ_repair_frozen_v1.json')
    completion=read('state/hybrid_READ_repair_complete_v1.json')
    if repair and completion:
        ledger=terminal_jobs(repair['performance_jobs'],completion['results'])
        add('adaptive_hybrid_READ_repair_terminal',ledger['passed'],ledger,
            'The original unsafe main result stays. All predeclared new seeds and controls require terminal outcomes.')
    else:add('adaptive_hybrid_READ_repair_terminal',False,None,'Do not ignore the recurrent main-test decode-tail symptom.')
    extension=read('configs/hybrid_READ_repair_layer_control_extension_v1.json')
    completion=read('state/hybrid_READ_repair_layer_control_complete_v1.json')
    if extension and completion:
        names=[n for c in extension['cells'] for n in c['jobs'].values()]
        ledger=terminal_jobs(names,completion['results'])
        add('repair_frozen_Layer_control_terminal',ledger['passed'],ledger,
            'Keep the added Layer control and its separate time block visible.')
    else:add('repair_frozen_Layer_control_terminal',False,None,'The all-four-method repair comparison includes its declared Layer control.')
    registration=read('configs/request_registration_deferral_frozen_v1.json')
    completion=read('state/request_registration_deferral_complete_v1.json')
    if registration and completion:
        ledger=terminal_jobs(registration['performance_jobs'],completion['results'])
        proofs=[read('state/'+r['gate']) for r in registration['qualification'].values()]
        add('request_registration_intervention_terminal',ledger['passed'] and all(p is not None for p in proofs),
            ledger,'Separate source overlay,proofs and adaptive follow-up. Failed correctness blocks performance,not evidence preservation.')
    else:add('request_registration_intervention_terminal',False,None,'Account for the new host-registration tail hypothesis and every declared control.')
    extension=read('configs/request_registration_q28_extension_frozen_v1.json')
    completion=read('state/request_registration_q28_extension_complete_v1.json')
    if extension and completion:
        ledger=terminal_jobs(extension['performance_jobs'],completion['results'])
        proofs=[read('state/'+r['gate']) for r in extension['qualification'].values()]
        add('request_registration_q28_extension_terminal',ledger['passed'] and all(p is not None for p in proofs),
            ledger,'Keep the other primary load and all frozen candidates. This is a separate adaptive extension.')
    else:add('request_registration_q28_extension_terminal',False,None,'Do not infer high-load safety from the q1.8 pilot.')
    bridge_repair=read('configs/request_registration_controlled_bridge_frozen_v1.json')
    completion=read('state/request_registration_controlled_bridge_complete_v1.json')
    if bridge_repair and completion:
        ledger=terminal_jobs(bridge_repair['performance_jobs'],completion['results'])
        proofs=[read('state/'+r['gate']) for r in bridge_repair['qualification'].values()]
        add('common_registration_controlled_bridge_terminal',ledger['passed'] and all(p is not None for p in proofs),
            ledger,'All original bridge conditions remain. Repaired continuous safety must not be inferred from Natural400 alone.')
    else:add('common_registration_controlled_bridge_terminal',False,None,'The original unsafe bridge also requires its declared common-runtime repair test.')

    for version in (1,2,3):
        name=f'decode_only_compile_correctness_frozen_v{version}.json'
        plan=read('configs/'+name)
        if not plan:
            add(f'decode_compile_v{version}_terminal',False,None,'Account for every declared role-local compiler test.')
            continue
        ledgers=[]
        for item in plan['qualification'].values():
            proof=read('state/'+item['gate'])
            ledger=terminal_jobs(item['jobs'],proof.get('stopped',[]) if proof else [])
            ledgers.append(dict(gate=item['gate'],proof_terminal=proof is not None,
                correctness_passed=proof.get('passed') if proof else None,ledger=ledger))
        add(f'decode_compile_v{version}_terminal',all(r['proof_terminal'] and r['ledger']['passed'] for r in ledgers),
            ledgers,'A terminal failed strict test blocks performance. Preserve the numerical mismatch and all NOT_RUN stages.')
    compile_performance=read('configs/decode_compile_performance_frozen_v1.json')
    if compile_performance:
        completion=read('state/decode_compile_performance_complete_v1.json')
        ledger=terminal_jobs(compile_performance['performance_jobs'],completion.get('results',[]) if completion else [])
        add('decode_compile_performance_terminal',completion is not None and ledger['passed'],ledger,
            'Do not close a declared paired compiler intervention before every frozen arm has a terminal outcome.')
    negative_control=read('configs/decode_compile_disabled_control_frozen_v1.json')
    if negative_control:
        ledger=terminal_jobs(negative_control['jobs'])
        outcome=read('state/decode_compile_disabled_control_terminal_v1.json')
        add('decode_compile_negative_controls_terminal',ledger['passed'] and outcome and outcome['complete'],
            ledger,'Distinguish old-teacher numerical non-equivalence from an introduced compiler or transport defect.')
    ir_plan=read('configs/decode_IR_priority_correctness_frozen_v1.json')
    if ir_plan:
        results=[]
        for item in ir_plan['qualification'].values():
            proof=read('state/'+item['gate'])
            results.append(dict(proof_terminal=proof is not None,gate=item['gate'],
                ledger=terminal_jobs(item['jobs'],proof.get('stopped',[]) if proof else [])))
        add('decode_IR_priority_correctness_terminal',all(r['proof_terminal'] and r['ledger']['passed'] for r in results),
            results,'Prospective exact-state and real mixed-row tests of the observed normalization-provider difference. No inherited performance.')
    scoped=read('configs/scoped_decode_compile_correctness_frozen_v1.json')
    if scoped:
        results=[]
        teacher=read('state/scoped_compile_teachers_complete_v1.json')
        results.append(dict(proof_terminal=teacher is not None,
            ledger=terminal_jobs(scoped['teachers'])))
        for item in scoped['qualification'].values():
            proof=read('state/'+item['gate'])
            results.append(dict(proof_terminal=proof is not None,gate=item['gate'],
                ledger=terminal_jobs(item['jobs'],proof.get('stopped',[]) if proof else [])))
        add('scoped_compile_correctness_terminal',all(r['proof_terminal'] and r['ledger']['passed'] for r in results),
            results,'Independent original-model stage teachers and new full-state,mixed32,Natural400 proofs. No inherited safety CI.')
    scoped_performance=read('configs/scoped_compile_confirmation_frozen_v1.json')
    if scoped_performance:
        completion=read('state/scoped_compile_confirmation_complete_v1.json')
        ledger=terminal_jobs(scoped_performance['performance_jobs'],completion.get('results',[]) if completion else [])
        controls=[read('state/'+r['gate']) for r in scoped_performance['qualification'].values()]
        add('scoped_compiler_paired_intervention_terminal',completion is not None and ledger['passed'] and all(p is not None for p in controls),
            ledger,'All21 frozen arms remain. Strict failures block new performance instead of inheriting old safety.')
    preflight=read('state/eager_v1_cold_dispatch_preflight_stop.json')
    add('unlaunched_eager_v1_preflight_accounted',preflight and preflight['GPU_starts']==0 and
        all(r['status']=='NOT_RUN' and r.get('reason') for r in preflight['results']),
        'state/eager_v1_cold_dispatch_preflight_stop.json',
        'The actual immutable cold-dispatch AST fails on CPU. Preserve all38 unrun configurations and the corrected V2 source.')
    eager=read('configs/eager_prefill_native_decode_correctness_frozen_v2.json')
    if eager:
        rows=[];teacher=read('state/'+eager['teacher_marker'])
        rows.append(dict(proof_terminal=teacher is not None,ledger=terminal_jobs(eager['teachers'])))
        for item in eager['qualification'].values():
            proof=read('state/'+item['gate'])
            rows.append(dict(proof_terminal=proof is not None,gate=item['gate'],
                ledger=terminal_jobs(item['jobs'],proof.get('stopped',[]) if proof else [])))
        add('eager_prefill_native_decode_correctness_terminal',all(r['proof_terminal'] and r['ledger']['passed'] for r in rows),
            rows,'Every prompt stage stays eager/CUDA. Full compiled native decode remains. Do not inherit a compiled-suffix teacher.')
    eager_performance=read('configs/eager_prefill_native_decode_confirmation_frozen_v2.json')
    if eager_performance:
        completion=read('state/eager_compile_confirmation_v2_complete_v1.json')
        ledger=terminal_jobs(eager_performance['performance_jobs'],completion.get('results',[]) if completion else [])
        proofs=[read('state/'+r['gate']) for r in eager_performance['qualification'].values()]
        add('eager_prefill_native_decode_confirmation_terminal',completion is not None and ledger['passed'] and all(p is not None for p in proofs),
            ledger,'The final bounded repair keeps all21 paired arms and the two Native references. A safety failure is not removed.')

    axis=read('analysis/axis_matrix_heldout_confirmation_r2.json')
    add('axis64_terminal',axis and len(axis['cells'])==64 and all(
        r['status'] not in ('NOT_RUN','PARTIAL') for r in axis['cells']),
        'analysis/axis_matrix_heldout_confirmation_r2.json','Invalid and CI-inconclusive cells remain visible.')
    iso=read('analysis/iso8_hybrid_heldout_confirmation_v1.json')
    add('iso8_terminal',iso and len(iso['cells'])==8 and all(
        r['paired_complete_starts']==3 or r['status']=='INVALID' for r in iso['cells']),
        'analysis/iso8_hybrid_heldout_confirmation_v1.json','C0 iso-work does not establish a C3 best-static win.')
    pipeline=read('analysis/proper_pipeline_load_discovery_v1.json')
    add('pipeline_supplement_terminal',pipeline and all(r['status']=='DISCOVERY_COMPLETE' for r in pipeline['cells']),
        'analysis/proper_pipeline_load_discovery_v1.json','Finite-trace N1 discovery is not sustainable capacity.')

    for name,file in (('credit_intervention','source_credit_host_intervention_v1.json'),
            ('completion_intervention','completion_guard_common_policy_intervention_v1.json')):
        value=read('analysis/'+file)
        add(name,value and value.get('complete',value.get('all_complete',False)),
            'analysis/'+file,'Controlled interventions preserve unfavorable outcomes. They do not assign exclusive GPU cost.')
    budget=read('analysis/scheduler_actual_budget_audit_v1.json')
    add('actual_BS_audit',budget and budget['passed'] and not budget['pending'],
        'analysis/scheduler_actual_budget_audit_v1.json','Every complete actual query batch is checked. Failed runs stay separate.')
    metrics=read('analysis/final_independent_client_metric_audit_v1.json')
    add('independent_metric_reconstruction',metrics and not metrics['pending'] and not metrics['missing_plans']
        and all(r['passed'] for r in metrics['results']),
        'analysis/final_independent_client_metric_audit_v1.json',
        'Reconstruct valid completed records. Planned failures stay terminal and have no invented latency.')
    for name,path in (('CPU_regression','analysis/cpu_final_regression_v1.json'),
            ('model_runtime_integrity','state/final_model_runtime_integrity_recheck.json')):
        value=read(path)
        valid=value is not None and value['passed']
        if valid and name=='CPU_regression':
            valid=value.get('source_hashes')=={p.name:digest(p) for p in SOURCE.iterdir() if p.suffix in ('.py','.cc','.cu','.h')}
        add(name,valid,path,'A CPU or hash pass does not replace serving correctness or decode safety. Stale CPU results do not qualify edited code.')
    add('paired_host_decomposition_present',read('analysis/final_C3_paired_host_service_decomposition_v1.json') is not None,
        'analysis/final_C3_paired_host_service_decomposition_v1.json','Same-ID host boundaries locate mean delay without inventing exclusive GPU shares.')
    protocol=read('analysis/final_protocol_integrity_audit_v1.json')
    add('frozen_protocol_and_GPU_UUID_audit',protocol and protocol['passed'],
        'analysis/final_protocol_integrity_audit_v1.json','Configuration and identity checks do not replace state or decode protection gates.')

    documents=[REPO/'poc_flashvep/reports'/name for name in DOCUMENTS]
    missing=[str(p.relative_to(REPO)) for p in documents if not p.exists()]
    add('required_reports_present',not missing,dict(missing=missing),
        'Presence alone is insufficient. The explicit root scientific review must bind the final report hashes.')
    review=read('state/scientific_closeout_review_v1.json')
    required_figures=[
        'dataset400_length_distribution','dataset400_input_output_scatter','dataset400_balance_audit',
        *['motivation400_confirmed_'+s for s in ('ttft_p95','itl_p99','tpot_p95','output_tokens','slo_goodput')],
        'axis400_heldout_ttft_ratio','axis400_heldout_goodput_ratio',
        'axis400_heldout_decode_itl_upper','axis400_heldout_decode_tpot_upper',
        'axis400_both_axes_decode_safe_winner_map',
        *['hybrid_iso8_confirmed_'+s for s in ('ttft','goodput','decode_gate')],
        'hybrid400_structural_DAG','hybrid400_structural_attention_distribution',
        'hybrid400_qualified_long_operator_cost','hybrid400_offline_replay_validation','hybrid400_bounded_offline_oracle',
        'qualified_projection_shape_efficiency',
        'controlled_output_and_load_counterfactuals','exploratory_minimal_demand_rule_comparison',
        'recovered_native_shared_ROI_activity_and_queues','recovered_native_observed_kernel_copy_timeline',
        'recovered_native_observed_kernel_gap_distribution',
        *['final_c3_main400_'+s for s in ('ttft_p95','itl_p99','tpot_p95','output_tokens','slo_goodput','joint_slo')],
        'final_c3_decode_noninferiority','final_c3_decode_noninferiority_common_policy',
        'credit_fixed_pool_sensitivity','continuation_chunk_residency_host_audit',
        *['controlled_C3_I3_bridge_'+s for s in ('ttft','itl','tpot','goodput')],
        'controlled_C3_I3_hybrid_token_runtime_bridge',
        *['adaptive_repair_confirmed_'+s for s in ('ttft','itl','tpot','goodput')],
        *['registration_repair_confirmed_'+s for s in ('ttft','itl','tpot','goodput','decode_gates')],
        *['registration_repair_q28_confirmed_'+s for s in ('ttft','itl','tpot','goodput','decode_gates')],
        *['registration_controlled_C3_I3_bridge_'+s for s in ('ttft','itl','tpot','goodput')],
        'registration_controlled_C3_I3_hybrid_token_runtime_bridge']
    eager_data=read('analysis/eager_compile_confirmation_v2_results_v1.json')
    if eager_data and any(r['starts']==3 for r in eager_data['arms'].values()):
        required_figures += ['eager_prefill_native_decode_'+s for s in ('ttft','goodput','noninferiority')]
    figure_results=[]
    omissions={r['stem']:r for r in review.get('omitted_figures',[])} if review else {}
    for stem in required_figures:
        files=[ROOT/'figures'/(stem+ext) for ext in ('.png','.pdf','_manifest.json')]
        ready=all(p.exists() and p.stat().st_size>0 for p in files)
        status='GENERATED' if ready else 'PENDING'
        if ready:
            meta=json.loads(files[-1].read_text())
            ready=bool(meta.get('evidence')) and bool(meta.get('caption'))
            status='GENERATED' if ready else 'INVALID_METADATA'
        elif stem in omissions:
            item=omissions[stem]
            ready=bool(item.get('reason')) and bool(item.get('evidence_path')) and (ROOT/item['evidence_path']).exists()
            if ready:status='SCIENTIFIC_STOP_REQUIRES_REVIEW'
        figure_results.append(dict(stem=stem,passed=ready,status=status))
    add('required_Figure_PNG_PDF_and_scope',all(r['passed'] for r in figure_results),figure_results,
        'Missing data require a reviewed scientific stop. Never create unmeasured curves to satisfy a filename check.')
    review_ok=review is not None and review.get('all_questions_answered') is True
    review_ok=review_ok and not review.get('unresolved_mandatory_work',[]) if review else False
    if review_ok:
        review_ok=all((REPO/r['path']).exists() and digest(REPO/r['path'])==r['SHA256']
            for r in review['reviewed_reports'])
        review_ok=review_ok and {r['question'] for r in review['answers']}==set(range(1,13))
        review_ok=review_ok and bool(review.get('explicit_not_run_and_stop_reasons'))
        catalog=ROOT/'analysis/figure_catalog_v1.json'
        review_ok=review_ok and catalog.exists() and digest(catalog)==review.get('figure_catalog_SHA256')
    add('scientific_closeout_review',review_ok,'state/scientific_closeout_review_v1.json',
        'Answer all twelve questions. Separate GO,HOLD,NO-GO,NOT_RUN and confidence limits. Bind reviewed reports by SHA256.')
    pending=[r['name'] for r in checks if not r['passed']]
    write(ROOT/'state/final_work_accounting.json',dict(time=time.time(),
        all_required_work_accounted=not pending,unresolved_mandatory_work=pending,checks=checks,
        scientific_success_not_implied=True,
        scope='Terminal failures and justified safety stops can complete a scientific campaign without a GO. '
            'Missing experiments,unreviewed evidence and unfinished controllers cannot. '
            'GPU restoration,lossless archive,commit,push and remote-SHA verification remain after this accounting.'))
    preserve()
    print('FINAL_WORK_ACCOUNTING',len(checks),'checks',len(pending),'pending',pending,flush=True)


if __name__=='__main__':
    main()
