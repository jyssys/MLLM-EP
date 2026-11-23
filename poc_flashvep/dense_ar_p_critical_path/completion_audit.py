"""Audit the full research scope before operational closeout."""
import argparse
import json
from pathlib import Path
import time
from .common import ROOT,SOURCE,write,digest,preserve
from .critical_analysis import csv_save
from .reports import NAMES


def read(relative):
    path=ROOT/relative
    return json.loads(path.read_text()) if path.exists() else None


def audit(final=False):
    preserve();requirements=[];inputs={};pending=[]
    def evidence(relative):
        value=read(relative)
        if value is not None:inputs[relative]=digest(ROOT/relative)
        return value
    def record(identifier,requirement,status,paths,detail):
        requirements.append(dict(id=identifier,requirement=requirement,status=status,evidence=paths,detail=detail))
        if status in ('MISSING','PENDING','UNVERIFIED'):pending.append(identifier)

    inventory=evidence('analysis/configuration_and_gate_inventory.json')
    metrics=evidence('analysis/raw_metric_recalculation.json');cpu=evidence('analysis/cpu_regression_audit.json')
    # CPU records use paths relative to the repository,not to the reports folder.
    from .common import REPO
    current={str(path.relative_to(REPO)):digest(path) for path in SOURCE.glob('*.py')}
    cpu_fresh=bool(cpu and cpu['passed'] and {row['path']:row['SHA256'] for row in cpu['source_files']}==current)
    record('R00','Preserve checkout,index,predecessor code and frozen datasets','PROVEN',
        ['state/preservation.json'],'preserve() checked the saved signatures and every frozen predecessor hash.')
    record('R01','Audit actual B/S,BI0,KV,model,UUID,source and execution status',
        'PROVEN' if inventory and inventory['passed'] and (not final or inventory['final']) else 'PENDING',
        ['analysis/configuration_and_gate_inventory.json'],'The inventory checks actual worker configs and composition proofs. Failed runs remain listed.')
    record('R02','Recompute raw client metrics and exact replay output work',
        'PROVEN' if metrics and metrics['passed'] and (not final or metrics['final']) else 'PENDING',
        ['analysis/raw_metric_recalculation.json'],'Independent reconstruction checks25 metrics,token events,SLO and cohort windows.')
    record('R03','CPU regression and current-source syntax','PROVEN' if cpu_fresh else 'PENDING',
        ['analysis/cpu_regression_audit.json'],'The audit requires the tested source hashes to equal the current source hashes.')
    gpu_budget=evidence('analysis/gpu_budget_accounting.json')
    record('R04','Account for every owned dispatch interval and the eight-GPU wall budget',
        'PROVEN_WITH_LIMITS' if gpu_budget and gpu_budget['passed'] and (not final or gpu_budget['final']) else 'PENDING',
        ['analysis/gpu_budget_accounting.json'],
        'Reserved process time includes startup and correctness controls. It is not GPU-exclusive compute time. Burner time is excluded.')
    historical=evidence('analysis/historical_stage_recalculation.json')
    record('R09','Recalculate published N3 host stages and retain the new R0 bridge',
        'PROVEN_WITH_LIMITS' if historical and historical['passed'] and len(historical.get('new_R0_bridges',[]))==6 else 'PENDING',
        ['analysis/historical_stage_recalculation.json'],
        'Historical raw recalculation is not a new GPU replication. The new R0 bridge has N1. New CPC and primary N3 are separate families.')

    families=(('R10','Independent N3 equal-output and Natural C3 credit intervention','c3_credit_confirmation',42),
        ('R11','Independent N3 P/D geometry intervention','p_geometry_confirmation',42),
        ('R12','Controlled16/64/128 and independent N3 Output192 stress','p_output_stress',42))
    for identifier,title,stem,count in families:
        path='analysis/'+stem+'_analysis.json';data=evidence(path)
        terminal=bool(data and len(data['execution_status'])==count and
            all(row['status'] in ('COMPLETE','INVALID_QUALITY','INVALID_RUNTIME','FAILED') for row in data['execution_status']))
        record(identifier,title,'PROVEN' if terminal else 'PENDING',[path],
            'Completion means every predeclared cell has a terminal status. Quality,decode and performance outcomes remain separate.')
    causal=evidence('analysis/p_critical_causal_summary.json')
    timeline=evidence('analysis/p_only_fulltrace_cpc_analysis.json')
    actual=[row for row in (timeline or {}).get('results',[]) if row.get('family')=='PUBLISHED_TRACE_REPRODUCTION_SAME_PAIR']
    terminal=bool(len(actual)==9 and all(row['evidence'] not in ('NOT_RUN_OR_INCOMPLETE',) for row in actual))
    record('R20','Same-request,same-pair P GPU/host CPC across all three traces',
        'PROVEN_WITH_LIMITS' if terminal else 'PENDING',
        ['analysis/p_only_fulltrace_cpc_analysis.json','analysis/p_critical_causal_summary.json'],
        'Qualified captures provide kernel union and path overlap. Failed observers remain diagnostic. Shared kernels are not assigned to individual requests.')
    record('R21','Exclusive causal GPU/credit/host attribution of the historical1.76s gap','PARTIAL',
        ['analysis/p_critical_causal_summary.json','analysis/scientific_verdict.json'],
        'New traces do not retroactively identify exclusive causal percentages. Credit pending intervals and DMA remain unmeasured. Root Cause is partial,not PASS.')
    serial=evidence('analysis/serial_serving_operator_analysis.json')
    serial_terminal=bool(serial and len(serial['results'])==12 and
        all(row['evidence'] not in ('NOT_RUN_OR_RUNNING',) for row in serial['results']))
    record('R22','Serial actual-API P/D and separate matched-work controls',
        'PROVEN_WITH_LIMITS' if serial_terminal else 'PENDING',['analysis/serial_serving_operator_analysis.json'],
        'Serial N3 best-fixed and N1 matched-work CPC are distinct families. D-first includes continuation and first-output work.')
    record('R23','Natural C3 D-exclusive operator fractions','NOT_MEASURED',
        ['analysis/serial_serving_operator_analysis.json'],
        'Qualified Natural C3 D operator attribution was not obtained. Serial fractions are not substituted.')

    norm=evidence('state/p_norm_ir_strict_complete.json')
    record('R30','Correctness-preserving P operator intervention gate',
        'FAILED_GATE_RETAINED' if norm else 'PENDING',['state/p_norm_ir_strict_complete.json'],
        'The P norm provider replacement failed exact KV/hidden checks. Primary performance was not run for that failed replacement.')
    for identifier,title,path in (
        ('R31','Coordinator evidence-write intervention','analysis/coordinator_delivery_diagnostic.json'),
        ('R32','Coordinator cyclic-GC intervention','analysis/coordinator_GC_diagnostic.json'),
        ('R33','Compiler artifact counterfactual audit','analysis/coordinator_stock_compiler_artifact_audit.json')):
        data=evidence(path)
        record(identifier,title,'DIAGNOSTIC_COMPLETED' if data else 'PENDING',[path],
            'Diagnostic evidence is not a validated online speedup or a complete explanation of the original1.85s tail.')
    conditional=(('R34','Coordinator composition confirmation','coordinator_hygiene_stable_stock_confirmation',
        ('state/coordinator_hygiene_stable_stock_not_run_v5.json',)),
        ('R35','Fixed-geometry D cap2→4 intervention','p_geometry_D_cap4_confirmation',
        ('state/p_geometry_D_cap4_not_run.json','state/p_geometry_D_cap4_confirmation_not_run.json')),
        ('R36','Common fixed-port coordinator confirmation','coordinator_hygiene_fixed_ports_confirmation',
        ('state/coordinator_hygiene_fixed_ports_not_run_v6.json',)),
        ('R39','Original warm-regime V8 confirmation dispatch decision','coordinator_hygiene_warm_regime_v8_confirmation',
        ('state/coordinator_hygiene_warm_regime_not_run_v8.json',)),
        ('R39B','V9 dispatch superseded before GPU timing','coordinator_hygiene_warm_regime_v9_confirmation',
        ('state/coordinator_hygiene_warm_regime_not_run_v9.json',)),
        ('R39D','Staged warm-regime independent confirmation','coordinator_hygiene_warm_regime_confirmation',
        ('state/coordinator_hygiene_warm_regime_not_run_v10.json',)),
        ('R13','Output192 D-cap-only controlled intervention','output192_D_cap_intervention',
        ('state/output192_D_cap_not_run.json',)))
    for identifier,title,stem,markers in conditional:
        data=evidence('analysis/'+stem+'_analysis.json');marker=next((evidence(path) for path in markers if (ROOT/path).exists()),None)
        statuses=data['execution_status'] if data else []
        def terminal_row(row):
            if row['status'] in ('COMPLETE','INVALID_QUALITY','INVALID_RUNTIME','FAILED','NOT_RUN_BY_RECORDED_DECISION'):return True
            config=evidence('configs/'+row['name']+'.json')
            proof=evidence('state/'+config['qualification_gate_v2']) if config else None
            return bool(proof and not proof['passed'])
        terminal=bool(statuses and all(terminal_row(row) for row in statuses))
        # A mode-specific NOT_RUN decision does not terminate unfinished modes.
        complete_stop=bool(marker and not marker.get('modes'))
        record(identifier,title,'PROVEN_WITH_LIMITS' if terminal else 'NOT_RUN_BY_GATE' if complete_stop else 'PENDING',
            ['analysis/'+stem+'_analysis.json',*markers],
            'A failed prerequisite is a preserved negative outcome. It is not waived or replaced by a weaker Stock baseline.')
    output_control=evidence('analysis/output192_D_cap_intervention_analysis.json')
    realization=evidence('analysis/output192_D_admission_realization.json')
    control_status=(output_control or {}).get('execution_status',[])
    realized_expected=sum(row['status'] in ('COMPLETE','INVALID_QUALITY','INVALID_RUNTIME') for row in control_status)
    realization_complete=bool(control_status and realization and
        all(row['status'] in ('COMPLETE','INVALID_QUALITY','INVALID_RUNTIME','FAILED','NOT_RUN_BY_RECORDED_DECISION')
            for row in control_status) and len(realization['rows'])==realized_expected)
    record('R13A','Check configured D cap against actual batches and registration waits',
        'PROVEN_WITH_LIMITS' if realization_complete else 'PENDING',
        ['analysis/output192_D_cap_intervention_analysis.json','analysis/output192_D_admission_realization.json'],
        'A cap4 setting is not a batch4 intervention. N1 histograms and host waits do not establish GPU-exclusive capacity or a serving win.')
    port_strict=evidence('state/fixed_port_stock_strict_complete_v6_reader_fix1.json')
    port_plan=evidence('configs/common_fixed_ports_qualification_v6.json')
    port_proofs=[evidence('state/'+row['gate']) for row in (port_plan or {}).get('items',[])]
    port_terminal=bool(port_strict and len(port_strict.get('jobs',[]))==3 and
        len(port_proofs)==12 and all(proof is not None for proof in port_proofs))
    record('R37','Test the compiler-cache port hypothesis and all twelve qualification outcomes',
        'PROVEN_WITH_LIMITS' if port_terminal else 'PENDING',
        ['state/fixed_port_stock_strict_complete_v6_reader_fix1.json','configs/common_fixed_ports_qualification_v6.json'],
        'Equal official cache roots did not restore exact KV/hidden equality. All8 greedy outputs matched in both comparisons. '
        'The repair is not sufficient. Baseline failures and unrun dependents remain explicit.')
    warm=evidence('state/official_warm_cache_stock_strict_complete_v7.json')
    record('R38','Prospective official warm-cache regime counterfactual',
        'CORRECTNESS_ONLY' if warm and ('error' in warm or len(warm.get('jobs',[]))==3) else 'PENDING',
        ['state/official_warm_cache_stock_strict_complete_v7.json','analysis/compiler_cache_regime_causal_audit.json'],
        'This new three-run regime test is not a retry of a failed GPU job. It does not erase the failed cold-control gate or establish a serving gain.')
    warm_plan=evidence('configs/common_warm_regime_qualification_v8.json')
    warm_proofs=[evidence('state/'+row['gate']) for row in (warm_plan or {}).get('items',[])]
    record('R39A','Fresh warm-regime8→32→400 qualification for all eight predeclared arms',
        'PROVEN_WITH_LIMITS' if len(warm_proofs)==8 and all(proof is not None for proof in warm_proofs) else 'PENDING',
        ['configs/common_warm_regime_qualification_v8.json'],
        'A successful warm strict triple is not itself a passed Natural400 composition. Every new arm retains its separate gate and failures.')
    corrected_plan=evidence('configs/common_warm_regime_qualification_v9.json')
    corrected_proofs=[evidence('state/'+row['gate']) for row in (corrected_plan or {}).get('items',[])]
    record('R39C','Corrected multi-cache reader and all preserved/reused qualification outcomes',
        'PROVEN_WITH_LIMITS' if len(corrected_proofs)==8 and all(proof is not None for proof in corrected_proofs) else 'PENDING',
        ['configs/common_warm_regime_qualification_v9.json'],
        'The CPU repair tracks every full and partial compiler region. It reuses completed GPU controls instead of repeating them. '
        'The old reader failures and every numerical failure remain preserved.')

    quality=evidence('analysis/native_relative_accuracy_analysis.json');arrival=evidence('analysis/arrival_trace_audit.json')
    record('R40','Native-relative output identity,task quality and frozen C1 gate',
        'PROVEN_WITH_LIMITS' if quality and len(quality['starts'])>=18 else 'PENDING',
        ['analysis/native_relative_accuracy_analysis.json'],
        'Exact token identity and task scores are separate. Context-cluster quality CIs are not server-start performance CIs.')
    record('R41','Identical Poisson arrivals and measured burst descriptors',
        'PROVEN_WITH_LIMITS' if arrival and len(arrival['traces'])==6 else 'PENDING',
        ['analysis/arrival_trace_audit.json'],'QPS is an offered mean rate. Burst associations have no arrival-pattern causal intervention.')
    verdict=evidence('analysis/scientific_verdict.json')
    record('R42','TTFT,goodput and both Native decode gates with no failed-cell exclusion',
        'PROVEN_WITH_LIMITS' if verdict and verdict['outcomes'] else 'PENDING',
        ['analysis/scientific_verdict.json'],'Scientific GO requires its declared gates. Audit completion does not require or imply a GO result.')
    record('R43','Future decode-safe predictor and active-decode insertion','NOT_RUN',
        ['analysis/scientific_verdict.json'],'No verified safe window exists. Unsafe insertion is prohibited,not silently treated as success.')

    active=evidence('attention_poc/state/active_protocol.json') or dict(completion='completion_v1.json')
    poc=evidence('attention_poc/state/'+active['completion']);poc_analysis=evidence('attention_poc/analysis/attention_head_poc_analysis.json')
    record('R50','User-added real-input attention-head PoC',
        'PROVEN_WITH_LIMITS' if poc and poc_analysis and poc.get('status') in ('COMPLETE','FAILED') else 'PENDING',
        ['attention_poc/state/'+active['completion'],'attention_poc/analysis/attention_head_poc_analysis.json'],
        'Isolated wall cost and tensor comparisons do not prove full-model quality,decode protection,or online gain.')
    dataset=evidence('analysis/dataset_reconstruction_audit.json')
    record('R60','Pinned-source byte-exact dataset reconstruction',
        'PROVEN' if dataset and dataset['passed'] else 'PENDING',['analysis/dataset_reconstruction_audit.json'],
        'Source rosters and templates reconstruct all six frozen datasets. Bulk third-party contexts remain local.')
    manifest=evidence('analysis/figure_manifest.json');regen=evidence('analysis/figure_regeneration_audit.json')
    requested_stems=('p_queue_kernel_overlap','p_request_critical_path_gantt','p_gpu_operator_resource_demand',
        'credit_retirement_equalreplay_pd_host_stages','credit_retirement_p_intervention',
        'credit_retirement_equalreplay_ttft_goodput','credit_retirement_equalreplay_stock_ratios',
        'credit_retirement_equalreplay_decode_protection','p_input_output_bottleneck_map')
    figures_ok=bool(manifest and set(requested_stems).issubset({row['stem'] for row in manifest['figures']})
        and all(digest(ROOT/item['path'])==item['SHA256']
        for row in manifest['figures'] for item in row['files'].values()))
    current_figures={item['path']:item['SHA256'] for row in (manifest or {}).get('figures',[]) for item in row['files'].values()}
    regen_current=bool(regen and regen['passed'] and regen.get('after')==current_figures)
    record('R61','Eight requested figure types with exact CSV,PNG and PDF',
        'PROVEN' if figures_ok and regen_current else 'PENDING',
        ['analysis/figure_manifest.json','analysis/figure_regeneration_audit.json'],
        'Hash equality verifies regeneration. The separate visual inspection must check labels,units,captions and clipped content.')
    visual=evidence('analysis/figure_visual_inspection.json')
    visual_ok=bool(visual and visual['passed'] and visual.get('manifest_SHA256')==digest(ROOT/'analysis/figure_manifest.json')
        and set(requested_stems).issubset({row['stem'] for row in visual.get('inspected',[])})
        and all(digest(ROOT/row['path'])==row['SHA256'] for row in visual.get('inspected',[])))
    record('R62','Visual inspection of rendered scientific figures',
        'PROVEN' if visual_ok else 'PENDING',['analysis/figure_visual_inspection.json'],
        'A file-presence test alone is insufficient for rendered artifacts.')
    drafts=all((ROOT/'report_drafts'/name).exists() for name in NAMES)
    record('R63','Six required reports with direct Q1–Q7 answers',
        'PROVEN_WITH_LIMITS' if drafts and '### Q7.' in (ROOT/'report_drafts'/NAMES[-1]).read_text() else 'PENDING',
        ['report_drafts/'+name for name in NAMES],
        'Final labels are written only after this audit. Final publication verifies all six resulting files separately.')
    pending_state=bool(inventory and (inventory['live'] or inventory['interrupted']))
    passed=final and not pending and not pending_state
    result=dict(time=time.time(),passed=passed,final=final,requirements=requirements,pending=pending,
        input_SHA256=inputs,scientific_root_cause='PARTIAL_ROOT_CAUSE_IDENTIFICATION',
        unresolved_scientific_requirements=[row['id'] for row in requirements if row['status'] in ('PARTIAL','NOT_MEASURED','NOT_RUN','FAILED_GATE_RETAINED')],
        scope='This audit checks campaign coverage,terminal evidence and honest limitations. It does not turn a failed scientific gate into a pass. '
            'PARTIAL,NOT_MEASURED and NOT_RUN remain explicit. Reports and conclusions must preserve them. '
            'Resource restoration,archive verification,push and remote SHA require later independent closeout checks.')
    write(ROOT/'state/science_completion_audit.json',result)
    csv_save(ROOT/'figures/requirement_completion_audit.csv',requirements)
    print('SCIENCE_COVERAGE_AUDIT',passed,'pending',pending,flush=True)
    if final:assert passed,result
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args();audit(args.final)
