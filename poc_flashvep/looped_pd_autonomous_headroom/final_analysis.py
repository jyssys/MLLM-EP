"""Produce measured tables and figures without merging different GPU starts."""
import csv
import math
import statistics
from .common import *
from .analysis_lane import quiet_analysis


def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def table(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT/name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'NOT_MEASURED' if v is None else json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def main():
    activate()
    from ..looped_pd_headroom_recovery.gates import actual_P_reference
    from .current_figures import main as refresh
    refresh()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = []
    def save(fig, name, scope):
        target = ROOT/'figures'/name
        fig.savefig(target.with_suffix('.png'), dpi=180)
        fig.savefig(target.with_suffix('.svg'))
        plt.close(fig)
        figures.append(dict(PNG='figures/'+name+'.png', SVG='figures/'+name+'.svg', scope=scope))
    ledgers = ('resume_joint_review', 'independent_joint_repeats', 'huginn_window_validation',
        'huginn_long_window', 'huginn_long_window_repeats', 'huginn_shorter', 'long_ouro_confirmation',
        'qualified_load_joint', 'qualified_lower_load_joint', 'profiler_version_control', 'profiler_prewarm_control')
    joints = []
    for ledger in ledgers:
        for r in load(ROOT/'state'/(ledger+'_results.json'), []):
            if not r.get('name') or 'gates' not in r:
                continue
            cfg = load(ROOT/'configs'/(r['name']+'.json'), {})
            g, q, p = r['gates'], r.get('queue_audit', {}), r.get('profile', {})
            strongest = actual_P_reference(cfg)
            final_ratio = g.get('P_completion_rate_in_interval', 0)/strongest if strongest else None
            interpretation = 'FINAL_FINITE_SCREEN_WITH_STATED_LIMITATIONS'
            if r['name'] == 'resume_huginn_s1_n256_nsys':
                interpretation = 'EXPLORATORY_ONLY_QUEUE_DEFINITION_SENSITIVE_NOT_FINAL_MODEL_PASS'
            joints.append(dict(name=r['name'], model=cfg.get('model'), input_tokens=cfg.get('length'),
                output_cap=cfg.get('cap'), offered=cfg.get('count'), nominal_QPS=cfg.get('arrival_rate'),
                GPUs=cfg.get('gpus'), arrival_seed=cfg.get('arrival_seed'),
                observation_seconds=g.get('observation_end', 0)-g.get('observation_start', 0),
                P_kernel_union_percent=100*g['P_actual_kernel_busy_central'] if g.get('P_actual_kernel_busy_central') is not None else None,
                P_service_proximity_percent=100*g['P_fraction_reference'] if g.get('P_fraction_reference') is not None else None,
                P_service_requests_per_s=g.get('P_completion_rate_in_interval'), P_queue_half_growth=q.get('half_growth'),
                strongest_final_finite_P_reference=strongest,
                P_service_proximity_final_percent=100*final_ratio if final_ratio is not None else None,
                final_reference_service_gate=final_ratio is not None and final_ratio >= .85,
                joint_screen_with_final_reference=bool(r.get('joint_screen_pass') and final_ratio is not None and final_ratio >= .85),
                P_queue_max=q.get('maximum'), original_phase_only_P_gate=g.get('P_compute_gate'), audited_P_gate=q.get('audited_P_gate'),
                D_TPOT_p95_ms=g.get('D_request_TPOT_p95_ms'), D_ITL_p99_ms=g.get('D_token_ITL_p99_central_ms'),
                D_max_ITL_whole_ms=g.get('D_token_ITL_max_whole_run_ms'), D_min_free_blocks=g.get('D_min_coordinator_free_blocks'),
                D_refusal_sample_fraction=g.get('D_credit_refusal_sample_fraction'),
                central_probes=r.get('central_probes'), exact_probe_output=r.get('correctness', {}).get('exact'),
                full_depth_work=r.get('correctness', {}).get('full_work'),
                profile_wall_difference_percent=100*p['matched_wall_difference_fraction'] if p.get('matched_wall_difference_fraction') is not None else None,
                profile_qualified=p.get('eligible_for_matched_clean'), joint_screen=r.get('joint_screen_pass'),
                accepted_fully_warmed_screen=r.get('accepted_joint_screen'),
                report_interpretation=interpretation,
                evidence='MEASURED_AND_TRACE_ANALYSIS',
                scope='Finite same-start capture. Kernel union is not SM occupancy. Original and audited queue gates remain separate.'))
    table('final_joint_screens.csv', joints)
    batch_rows = []
    for row in load(ROOT/'state/batch_capacity_control_results.json', []):
        for cohort in row.get('cohorts', []):
            value = dict(cohort, name=row['name'], model=row['model'], full_run_complete=row['complete'])
            value['finite_P_requests_per_s'] = cohort['batch']/cohort['P_service_s'] if cohort.get('P_service_s') else None
            batch_rows.append(value)
    table('final_stronger_P_batch_capacity.csv', batch_rows)
    repeated = [r for r in joints if r['name'].startswith('qualified_ouro_s')]
    uncertainty = []
    for key in ('P_kernel_union_percent', 'P_service_requests_per_s', 'D_TPOT_p95_ms', 'D_ITL_p99_ms'):
        values = [r[key] for r in repeated if isinstance(r.get(key), (int, float))]
        if len(values) == 3:
            mean = statistics.mean(values)
            radius = 4.302652729911275*statistics.stdev(values)/math.sqrt(3)
            uncertainty.append(dict(metric=key, starts=3, mean=mean, minimum=min(values), maximum=max(values),
                student_t_95_low=mean-radius, student_t_95_high=mean+radius,
                scope='Independent-start t interval, df=2. No clipping. Three starts give weak population inference. Tokens are not independent starts.'))
    table('final_independent_start_uncertainty.csv', uncertainty)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for ax, key, threshold, title in zip(axes,
            ('P_kernel_union_percent', 'P_service_proximity_final_percent', 'D_ITL_p99_ms'),
            (80, 85, 65.08287642151119), ('P kernel union (%)', 'P service / strongest final reference (%)', 'D ITL p99 (ms)')):
        ax.bar(range(1, len(repeated)+1), [r[key] for r in repeated], color='#1769aa')
        ax.axhline(threshold, color='#aa3333', linestyle='--', label='Frozen screen')
        ax.set(title=title, xlabel='Independent Ouro start', xticks=list(range(1, len(repeated)+1)))
        ax.legend(fontsize=8)
    fig.suptitle('Ouro 4096/32: three same-window screens with real D-local Prefill\nFinite cohorts. Kernel union is not physical SM saturation.', fontsize=11)
    save(fig, 'final_ouro_independent_joint_screens', 'Each column uses the same three original runs. No P/D start stitching.')
    agg = load(ROOT/'state/matched_aggregated_results.json', [])
    pairs = [r for r in agg if isinstance(r.get('PD'), dict) and isinstance(r.get('DP'), dict)]
    aggregate_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    metrics = ('TTFT_ms_p95', 'request_TPOT_ms_p95', 'token_ITL_ms_p99', 'total_tokens_per_s')
    for index, model in enumerate(('ouro', 'huginn')):
        selected = sorted([r for r in pairs if r['model'] == model], key=lambda r: r['target_QPS'])
        for arm, label in (('PD', 'Static PD'), ('DP', 'Native Mixed DP2')):
            for row in selected:
                aggregate_rows.append({**row[arm], 'model': model, 'arm': label, 'target_QPS': row['target_QPS'],
                    'paired_parity': row.get('paired_output_parity', {}).get('passed')})
            for ax, metric in zip(axes[index], metrics):
                if selected:
                    ax.plot([r['target_QPS'] for r in selected], [r[arm][metric] for r in selected], 'o-', label=label)
                else:
                    ax.text(.5, .5, 'NOT_MEASURED', ha='center', transform=ax.transAxes)
                ax.set(title=model+' / '+metric, xlabel='Offered QPS')
                if selected:
                    ax.legend(fontsize=8)
    fig.suptitle('Matched finite serving controls: one independent start per rate\nTwo-chunk calibration only. Not a sustainable-capacity estimate.', fontsize=12)
    save(fig, 'final_duetserve_style_controls', 'Actual paired rate controls. Missing cells remain NOT_MEASURED.')
    table('final_aggregated_controls.csv', aggregate_rows)
    curve = load(ROOT/'state/qualified_probe_curve_results.json', [])
    curve_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, ax, threshold in zip(('request_TPOT_ms_p95', 'token_ITL_ms_p99'), axes,
            (48.3363260853697, 65.08287642151119)):
        valid = [r for r in curve if r.get('probe')]
        if valid:
            ax.plot([r['tokens'] for r in valid], [r['probe'][key] for r in valid], 'o-', label='Probe ON')
            ax.plot([r['tokens'] for r in valid], [r['baseline'][key] for r in valid], 's--', label='Matched OFF')
        else:
            ax.text(.5, .5, 'NO COMPLETED SIZE COMPARISON', ha='center', transform=ax.transAxes)
        ax.axhline(threshold, color='#aa3333', linestyle=':', label='Frozen D screen')
        ax.set(xlabel='Tokens in each independent D-local prompt', ylabel=key+' (ms)')
        ax.legend(fontsize=8)
    for row in curve:
        probe_name = row.get('probe', {}).get('name')
        probe_path = ROOT/'raw'/str(probe_name)/'D_probe_requests.jsonl'
        completed = [json.loads(line) for line in probe_path.read_text().splitlines()] if probe_path.exists() else []
        durations = [1000*(r['finish']-r['time']) for r in completed]
        curve_rows.append(dict(tokens=row['tokens'], status=row.get('status'), accepted=row.get('accepted'),
            exact=row.get('correctness', {}).get('exact'), full_work=row.get('correctness', {}).get('full_work'),
            completed_probes=len(completed), additional_prefill_recurrent_rows=len(completed)*row['tokens']*4,
            probe_host_duration_min_ms=min(durations) if durations else None,
            probe_host_duration_max_ms=max(durations) if durations else None,
            scope='Ouro R4 independent local prompts. Host durations include waiting and are not isolated GPU times.',
            **{side+'_'+metric: row.get(side, {}).get(metric) for side in ('baseline', 'probe')
               for metric in ('request_TPOT_ms_p95', 'token_ITL_ms_p99', 'token_ITL_ms_max', 'elapsed_s')}))
    fig.suptitle('Ouro: actual D-local Prefill size versus Decode tails\nOne matched start per size. These are not migrated continuations.', fontsize=11)
    save(fig, 'final_safe_prefill_probe_curve', 'The sweep stops at its first failed gate. No larger untested size is marked safe.')
    table('final_probe_size_curve.csv', curve_rows)
    loads = load(ROOT/'state/qualified_probe_load_results.json', [])
    load_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for key, ax, threshold in zip(('request_TPOT_ms_p95', 'token_ITL_ms_p99'), axes,
            (48.3363260853697, 65.08287642151119)):
        valid = [r for r in loads if r.get('probe')]
        if valid:
            for side in ('baseline', 'probe'):
                ax.plot([r['additional_prompt_tokens_per_response_wall_s'] for r in valid],
                    [r[side][key] for r in valid], 'o-', label=side)
        else:
            ax.text(.5, .5, 'NOT_MEASURED', ha='center', transform=ax.transAxes)
        ax.axhline(threshold, color='#aa3333', linestyle=':', label='Frozen D screen')
        ax.set(xlabel='Actual added prompt tokens / response-wall second', ylabel=key+' (ms)')
        ax.legend(fontsize=8)
    for row in loads:
        load_rows.append({k: row.get(k) for k in ('name', 'tokens', 'requested_interval_s',
            'completed_probes', 'additional_prefill_recurrent_rows',
            'additional_prompt_tokens_per_response_wall_s', 'accepted', 'status', 'scope')} |
            {side+'_'+metric: row.get(side, {}).get(metric) for side in ('baseline', 'probe')
             for metric in ('request_TPOT_ms_p95', 'token_ITL_ms_p99', 'token_ITL_ms_max', 'elapsed_s')})
    fig.suptitle('Ouro: exact D-local Prefill load versus Decode tails\nActual admitted work, not requested rate. One matched start per level.', fontsize=11)
    save(fig, 'final_safe_prefill_probe_load', 'No simultaneous P device capture. This is a Decode-load sensitivity control, not Dynamic A speedup.')
    table('final_probe_load_curve.csv', load_rows)
    with (ROOT/'completed_serving_current.csv').open() as handle:
        metrics_by_name = {r['name']: r for r in csv.DictReader(handle)}
    paired = []
    for start in (1, 2, 3):
        off = metrics_by_name.get(f'qualified_ouro_s{start}_baseline128')
        on = metrics_by_name.get(f'qualified_ouro_s{start}_probe128')
        if not off or not on:
            continue
        for metric in ('request_TPOT_ms_p95', 'token_ITL_ms_p99', 'token_ITL_ms_max', 'elapsed_s'):
            paired.append(dict(model='ouro', start=start, metric=metric, baseline=float(off[metric]),
                probe=float(on[metric]), paired_difference=float(on[metric])-float(off[metric]),
                scope='Same requests and arrival trace within each start. Positive latency difference is worse.'))
    table('final_paired_probe_effect.csv', paired)
    paired_ci = []
    for metric in {r['metric'] for r in paired}:
        values = [r['paired_difference'] for r in paired if r['metric'] == metric]
        if len(values) == 3:
            mean = statistics.mean(values)
            radius = 4.302652729911275*statistics.stdev(values)/math.sqrt(3)
            paired_ci.append(dict(metric=metric, starts=3, mean_difference=mean,
                student_t_95_low=mean-radius, student_t_95_high=mean+radius,
                scope='Paired start-level t interval, df=2. Not a request-independent or token-independent interval.'))
    table('final_paired_probe_uncertainty.csv', paired_ci)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, model in zip(axes, ('ouro', 'huginn')):
        names = [f'bulk_written_{model}_{variant}_clean24' for variant in ('eager', 'graph', 'bulk')]
        values = [metrics_by_name.get(n, {}).get('elapsed_s') for n in names]
        if all(v not in (None, 'NOT_MEASURED') for v in values):
            ax.bar(('Eager', 'Native graph', 'Graph + host range'), list(map(float, values)), color='#3b8067')
        ax.set(title=model, ylabel='Arrival-to-drain wall time (s)')
    fig.suptitle('Matched 24-request runtime controls: one start per method\nHost wall time, not exclusive GPU execution time.', fontsize=11)
    save(fig, 'final_generic_runtime_control', 'Actual matched useful work. Do not attribute changed-workload kernel percentages to this control.')
    # These are separate D captures. Never label them as the P-capture run.
    for record in load(ROOT/'state/decode_timeline_results.json', []):
        profile = record.get('profile', {})
        if profile.get('status') != 'PARSED':
            continue
        folder = ROOT/'raw'/record['name']
        kernels = load(folder/'aligned_kernels.json')['kernels']
        left, right = profile['window_start_monotonic'], profile['window_end_monotonic']
        from ..looped_pd_headroom_recovery.profile_analysis import union
        bins, cursor = [], left
        while cursor < right:
            end = min(right, cursor+.25)
            spans = [(max(cursor, k['start_monotonic']), min(end, k['end_monotonic']))
                for k in kernels if k['end_monotonic'] > cursor and k['start_monotonic'] < end]
            bins.append(dict(start_s=cursor-left, end_s=end-left, D_kernel_union_percent=100*union(spans)/(end-cursor)))
            cursor = end
        table(record['name']+'_kernel_bins.csv', bins)
        fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
        ax.stairs([b['D_kernel_union_percent'] for b in bins],
            [b['start_s'] for b in bins]+[bins[-1]['end_s']], fill=True, alpha=.6)
        ax.set(xlabel='Seconds from actual D capture start', ylabel='D kernel union (%)', ylim=(0, 105))
        fig.suptitle(record['model']+' D-only capture: '+('qualified' if profile['eligible_for_matched_clean'] else 'PERTURBED DIAGNOSTIC')+
            '\nSeparate start. Not simultaneous P/D device proof or guaranteed usable idle.', fontsize=11)
        save(fig, 'final_'+record['name'], record['scope'])
    memory = []
    totals = dict(line.split(', ') for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'], text=True).splitlines())
    for name in [f'qualified_ouro_s{i}_probe128' for i in (1, 2, 3)]+['window_huginn_s1_probe384', 'short_huginn_s1_probe384']:
        cfg = load(ROOT/'configs'/(name+'.json'))
        if not cfg:
            continue
        for path in (ROOT/'raw'/name).glob('memory_inventory_initial_*.json'):
            value = load(path)
            role = value.get('role')
            if role not in ('prefill', 'decode'):
                continue
            gpu = cfg['gpus'][0 if role == 'prefill' else 1]
            total = float(totals[str(gpu)])*1024**2
            for key, amount in value.items():
                if key.endswith('_bytes') and isinstance(amount, (int, float)):
                    memory.append(dict(name=name, model=cfg['model'], role=role, physical_GPU=gpu, component=key,
                        bytes=amount, GiB=amount/1024**3, percent_physical_HBM=100*amount/total,
                        scope='Initial unique-storage inventory. Allocator total overlaps its components. Free KV blocks are logical pool credit, not unallocated HBM.'))
    table('final_memory_inventory_GiB_percent.csv', memory)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    component_names = ('weight_and_model_buffer_storage_bytes', 'KV_storage_bytes',
        'exposed_runner_workspace_storage_bytes', 'exposed_NIXL_hidden_staging_storage_bytes',
        'exposed_graph_tensor_storage_bytes', 'opaque_or_unattributed_allocator_bytes')
    labels = ('Weights/model buffers', 'KV pool', 'Exposed workspace', 'NIXL staging', 'Exposed graphs', 'Allocator residual')
    for ax, model, name in zip(axes, ('ouro', 'huginn'),
            ('qualified_ouro_s1_probe128', 'window_huginn_s1_probe384')):
        bottom = [0., 0.]
        for component, label in zip(component_names, labels):
            values = [sum(r['GiB'] for r in memory if r['name'] == name and r['role'] == role
                and r['component'] == component) for role in ('prefill', 'decode')]
            ax.bar(('P', 'D'), values, bottom=bottom, label=label)
            bottom = [a+b for a, b in zip(bottom, values)]
        ax.axhline(float(totals['0'])/1024, color='black', linestyle='--', label='Physical HBM')
        ax.set(title=model, ylabel='Initial unique-storage GiB', ylim=(0, 85))
    axes[1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1., 1.))
    fig.suptitle('HBM allocation inventory before measurement\nKV pool allocation is not active KV usage. Peak device use is reported separately.', fontsize=11)
    save(fig, 'final_hbm_component_inventory', 'Non-overlapping unique-storage components. Not complete graph-pool attribution or peak HBM.')
    peak_memory = []
    for name, row in metrics_by_name.items():
        cfg = load(ROOT/'configs'/(name+'.json'), {})
        for index, role in enumerate(('P', 'D')):
            value = row.get(role+'_sampled_peak_HBM_bytes')
            if value not in (None, '', 'NOT_MEASURED') and len(cfg.get('gpus', [])) > index:
                gpu = cfg['gpus'][index]
                amount = float(value)
                peak_memory.append(dict(name=name, role=role, physical_GPU=gpu, GiB=amount/1024**3,
                    percent_physical_HBM=100*amount/(float(totals[str(gpu)])*1024**2),
                    scope='Sampled device HBM peak, not a continuous allocation maximum. DP columns identify replicas.'))
    table('final_sampled_peak_HBM_GiB_percent.csv', peak_memory)
    manifest = []
    for path in sorted((ROOT/'configs').glob('*.json')):
        cfg = load(path, {})
        if not isinstance(cfg, dict) or 'name' not in cfg:
            continue
        folder = ROOT/'raw'/cfg['name']
        result = load(folder/'dispatcher_result.json', {})
        watchdog = load(folder/'watchdog.json', {})
        status = 'COMPLETE' if result.get('complete') else 'FAILED_OR_PARTIAL' if folder.exists() else 'NOT_RUN'
        if cfg.get('capacity_mode') and folder.exists() and not (folder/'cohorts.jsonl').exists():
            status = 'INVALID_CAPACITY_DISPATCH' if cfg.get('worker') != 'capacity_worker' else status
        manifest.append(dict(name=cfg['name'], model=cfg.get('model'), status=status, watchdog=watchdog,
            configuration_SHA256=digest(path), source_archive=str((folder/'campaign_source.zip').relative_to(ROOT)),
            count=cfg.get('count'), input_tokens=cfg.get('length'), output_cap=cfg.get('cap'), GPUs=cfg.get('gpus')))
    table('final_execution_manifest.csv', manifest)
    write(ROOT/'final_data_summary.json', dict(time=time.time(), joint_screens=joints, independent_start_uncertainty=uncertainty,
        paired_probe_uncertainty=paired_ci,
        aggregate_pairs=len(pairs), probe_size_results=curve_rows, execution_counts={key: sum(r['status'] == key for r in manifest)
            for key in ('COMPLETE', 'FAILED_OR_PARTIAL', 'NOT_RUN', 'INVALID_CAPACITY_DISPATCH')},
        limitations=['Physical SM and bandwidth counters are NOT_MEASURED.',
            'Original-HF fidelity and downstream accuracy are NOT_MEASURED.',
            'Tiny D-local probes do not establish migrated continuation speedup.',
            'Finite overloaded cohorts do not establish sustainable capacity.']))
    write(ROOT/'final_figure_manifest.json', dict(time=time.time(), figures=figures,
        same_window_manifests=[str(p.relative_to(ROOT)) for p in (ROOT/'raw').glob('*/same_window_figure_manifest.json')],
        scope='Every figure identifies its actual measurement scope. A host request span is not a GPU kernel timeline.'))
    print(json.dumps(dict(joint_screens=len(joints), executions=len(manifest), figures=len(figures))), flush=True)


if __name__ == '__main__':
    with quiet_analysis('final_evidence_analysis'):
        main()
