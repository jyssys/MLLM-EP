"""Plot accepted measurements. Do not manufacture missing device evidence."""
import csv
from .common import *


def main():
    initialize()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none'})
    target = ROOT / 'figures'
    target.mkdir(exist_ok=True)
    manifest = []
    rows = []
    qualification = {}
    marker_path = ROOT / 'state/measurement_validation.json'
    markers = json.loads(marker_path.read_text()).get('markers', {}) if marker_path.exists() else {}
    for name in ('primary_screen', 'transport_screen', 'fine_suffix', 'static_campaign', 'dynamic_campaign', 'confirmation'):
        path = ROOT / 'state' / (name + '.json')
        if not path.exists():
            continue
        ledger = json.loads(path.read_text())
        candidates = list(ledger.get('results', []))
        candidates += [row for model in ledger.get('models', {}).values()
                       for row in model.get('serving', [])]
        candidates += [row for model in ledger.get('models', {}).values()
                       for row in model.get('results', [])]
        for row in candidates:
            if row.get('name'):
                qualification[row['name']] = row.get('passed', False)
    for path in (ROOT / 'raw').glob('*/serving_metrics.json'):
        cfg = json.loads((ROOT / 'configs' / (path.parent.name + '.json')).read_text())
        if cfg.get('state_probe') or cfg.get('verify_transfer') or cfg.get('calibration_serial'):
            continue
        if cfg.get('kernel_activity_seconds'):
            continue
        row = json.loads(path.read_text())
        row['output_work_lifetime_gate'] = qualification.get(path.parent.name)
        row['main_screen_gate_scope'] = ('Reviewed main/confirmation row' if path.parent.name in qualification
            else 'Not a main-screen row. This is not a failed correctness verdict.')
        if cfg['arm'] == 'PD_G1':
            method = 'Native PD' if cfg.get('atomic_prompt') else 'Staged Native PD'
        elif not cfg.get('dynamic') and cfg.get('axis') == 'loop' and cfg.get('cut') == cfg['depth']:
            method = 'Full Snapshot (Loop connector)'
        elif not cfg.get('dynamic') and cfg.get('axis') == 'token' and cfg.get('prefix_tokens') == cfg['length']:
            method = 'Full Snapshot (Token connector)'
        else:
            method = cfg['arm']
        row.update(axis=cfg.get('axis'), dynamic=cfg.get('dynamic', False),
                   cut=cfg.get('cut'), prefix=cfg.get('prefix_tokens'), load=cfg.get('load'),
                   display_method=method, count=cfg['count'],
                   independent_start=cfg.get('independent_start'),
                   source=str(path.relative_to(ROOT)), source_SHA256=digest(path))
        work_path = path.parent / 'split_work_and_payload.json'
        if work_path.exists():
            work = json.loads(work_path.read_text())
            for field in ('completed_offloaded_serving_requests',
                          'assigned_D_recurrent_token_rows_on_completed_requests',
                          'measured_transport_payload_bytes', 'measured_snapshot_hidden_bytes'):
                row[field] = work[field]
        marker_plan = None
        if cfg['arm'] == 'PD_G1' and cfg.get('atomic_prompt'):
            marker_plan = 'native'
        elif not cfg.get('dynamic') and cfg['arm'] in ('Token_Static', 'Loop_Static'):
            marker_plan = ('token' + str(cfg['prefix_tokens']) if cfg['axis'] == 'token'
                           else 'loop' + str(cfg['cut']))
        control = markers.get(cfg['model'], {}).get(marker_plan)
        row.update(event_marker_control_status=('PASS_MATCHED_HEAVY_PAIR' if control.get('passed')
            else 'FAIL_MATCHED_HEAVY_PAIR') if control and 'wall_change_fraction' in control else 'NOT_MEASURED',
            event_marker_wall_change_fraction=control.get('wall_change_fraction') if control else None,
            event_marker_control_scope='One heavy-load geometry pair only. Not a per-run, per-load, or population overhead bound.')
        rows.append(row)
    columns = sorted({key for row in rows for key, value in row.items()
                      if not isinstance(value, (dict, list))})
    if rows:
        with (ROOT / 'serving_results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)

    def save(fig, name, selected, scope):
        name = name.lower()
        fig.tight_layout()
        files = []
        for extension in ('png', 'svg'):
            path = target / (name + '.' + extension)
            fig.savefig(path, dpi=180, bbox_inches='tight')
            files.append(str(path.relative_to(ROOT)))
        plt.close(fig)
        manifest.append(dict(name=name, status='MEASURED_WITH_STATED_INSTRUMENTATION', files=files, scope=scope,
            sources={row['source']: row['source_SHA256'] for row in selected}))

    for model in ('ouro', 'huginn'):
        measured = [row for row in rows if row['model'] == model and row['output_work_lifetime_gate']
                    and row['count'] == 128 and row['independent_start'] == 1]
        if not measured:
            for figure in ('motivation', 'TTFT_decomposition', 'static_splits', 'dynamic_methods',
                           'matched_work', 'SLO_goodput_and_tails'):
                manifest.append(dict(name=f'{model}_{figure}', status='NOT_MEASURED'))
            continue
        for load in ('low', 'medium', 'heavy'):
            selected = [row for row in measured if row['load'] == load]
            if not selected:
                continue
            labels = [row['display_method'] + (' k' + str(row['prefix']) if row['axis'] == 'token'
                      else ' R' + str(row['cut'])) for row in selected]
            fig, axes = plt.subplots(1, 3, figsize=(max(10, len(selected) * .8), 4.2))
            for ax, field, title in zip(axes,
                    ('TTFT_p95_ms', 'queue_ms_p95', 'service_TTFT_ms_p95'),
                    ('Request TTFT p95', 'Queue / admission p95', 'Service TTFT p95')):
                ax.bar(range(len(selected)), [row[field] / 1000 for row in selected], color='#3977a9')
                ax.set_xticks(range(len(selected)), labels, rotation=75, ha='right')
                ax.set(title=title, ylabel='Seconds')
                ax.grid(axis='y', alpha=.2)
            fig.suptitle(f'{model.title()} | {load} | event-marked serving | quantiles are not additive', y=1.03)
            save(fig, f'{model}_{load}_TTFT_decomposition', selected,
                 'Per-request decomposition precedes quantiles. Queue and service p95 values are separate panels, not a stacked TTFT p95.')

            comparison = []
            for method in ('Native PD', 'Token_Static', 'Token_Dynamic', 'Loop_Static', 'Loop_Dynamic'):
                choices = [row for row in selected if row['display_method'] == method]
                if method.endswith('_Static'):
                    choices = [row for row in choices if row['Decode_SLO_screen']]
                best = max(choices, key=lambda row: (row['joint_SLO_goodput'],
                    -row['TTFT_p95_ms']), default=None)
                comparison.append((method, best))
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            fields = ('TTFT_p95_ms', 'service_TTFT_ms_p95', 'E2E_p95_ms',
                      'joint_SLO_goodput', 'TPOT_p95_ms', 'ITL_p99_ms')
            titles = ('Request TTFT p95', 'Service TTFT p95', 'E2E p95',
                      'Joint-SLO goodput', 'Request-average TPOT p95', 'Delivery ITL p99')
            labels = ['Native PD', 'Static Token', 'Dynamic Token', 'Static Loop', 'Dynamic Loop']
            for index, (_, row) in enumerate(comparison):
                if row is not None and index > 0 and 'completed_offloaded_serving_requests' in row:
                    labels[index] += f"\n{row['completed_offloaded_serving_requests']}/{row['count']} actual offloads"
            actual = [row for _, row in comparison if row is not None]
            for ax, field, title in zip(axes.flat, fields, titles):
                for index, (method, row) in enumerate(comparison):
                    if row is None:
                        measured_unsafe = method.endswith('_Static') and any(
                            candidate['display_method'] == method for candidate in selected)
                        missing = 'NO DECODE-SAFE\nPARTIAL CUT' if measured_unsafe else 'NOT MEASURED'
                        ax.text(index, .02, missing,
                                rotation=90, ha='center', va='bottom', fontsize=6,
                                transform=ax.get_xaxis_transform())
                    else:
                        ax.bar(index, row[field], color='#3977a9' if index in (0, 3, 4) else '#c86b35')
                ax.set_xticks(range(5), labels, rotation=35, ha='right')
                ax.set_xlim(-.5, 4.5)
                ax.set(title=title, ylabel='Requests/s' if field == 'joint_SLO_goodput' else 'Milliseconds')
                ax.grid(axis='y', alpha=.2)
            frozen = selected[0]['frozen_SLO']
            axes[1, 1].axhline(.9 * frozen['TPOT_ms'], color='black', ls='--')
            axes[1, 2].axhline(frozen['ITL_ms'], color='black', ls='--')
            fig.suptitle(f'{model.title()} | {load} | event-marked, one-start exploratory comparison\n'
                'Partial Static cuts selected on this cohort. Dynamic actions use observed state only.', y=1.02)
            save(fig, f'{model}_{load}_five_method_comparison', actual,
                'Static selection maximizes observed joint goodput among measured Decode-safe cuts, then minimizes TTFT. '
                'This is cohort selection, not held-out confirmation or a static oracle. Missing methods remain explicit. '
                'Dynamic output parity does not imply actual offload or a passing Decode tail.')

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for arm in sorted({row['display_method'] for row in measured}):
            group = [row for row in measured if row['display_method'] == arm]
            boundaries = sorted({(row['cut'], row['prefix']) for row in group}, key=str)
            for cut, prefix in boundaries:
                values = sorted((row for row in group if row['cut'] == cut and row['prefix'] == prefix),
                                key=lambda row: row['nominal_QPS'])
                label = arm if any(row['dynamic'] for row in values) else arm + (
                    ' k' + str(prefix) if prefix is not None else ' R' + str(cut))
                for ax, field in zip(axes, ('joint_SLO_goodput', 'TPOT_p95_ms', 'ITL_p99_ms')):
                    ax.plot([row['nominal_QPS'] for row in values], [row[field] for row in values],
                            marker='o', ms=3, label=label)
        axes[0].set(ylabel='Joint-SLO requests/s', title='Finite measured SLO-goodput')
        axes[1].set(ylabel='Milliseconds', title='Request-average TPOT p95')
        axes[2].set(ylabel='Milliseconds', title='Token ITL p99')
        slo = measured[0]['frozen_SLO']
        axes[1].axhline(.9 * slo['TPOT_ms'], color='black', ls='--', label='Frozen 90% TPOT margin')
        axes[2].axhline(slo['ITL_ms'], color='black', ls='--', label='Frozen ITL p99 limit')
        for ax in axes:
            ax.set_xlabel('Nominal offered requests/s')
            ax.grid(alpha=.2)
        axes[-1].legend(fontsize=6, loc='upper left', bbox_to_anchor=(1.02, 1))
        save(fig, model + '_SLO_goodput_and_tails', measured,
             'Actual finite cohorts. Nominal QPS is the x-axis. Realized arrival QPS is retained in the CSV. No stable-capacity or physical-saturation claim.')

        selected = [row for row in measured if row['load'] == 'heavy' and row['arm'] != 'PD_G1']
        if selected:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for axis, color in (('token', '#c86b35'), ('loop', '#3977a9')):
                group = [row for row in selected if row['axis'] == axis and not row['dynamic']]
                labels = [str(row['prefix'] if axis == 'token' else row['cut']) for row in group]
                for ax, field in zip(axes, ('service_TTFT_ms_p95', 'P_prompt_device_enclosure_ms_p95',
                                           'D_continuation_to_first_token_ms_p95')):
                    values = [(label, row[field]) for label, row in zip(labels, group) if row[field] is not None]
                    if values:
                        ax.plot([axis + ':' + label for label, _ in values], [v for _, v in values],
                                marker='o', color=color, label=axis)
            for ax, title in zip(axes, ('Service TTFT p95', 'P device enclosure p95', 'D continuation-to-first-output p95')):
                ax.set(title=title, ylabel='Milliseconds')
                ax.tick_params(axis='x', rotation=70)
                ax.grid(alpha=.2)
            save(fig, model + '_static_split_granularity', selected,
                 'Separate Token and Loop boundary labels. Device enclosures include launch gaps and interleaved requests. They are not exclusive kernel time.')
        activity = ROOT / 'activity_figure_manifest.json'
        recorded = json.loads(activity.read_text()).get('figures', []) if activity.exists() else []
        actual = [row for row in recorded if row.get('model') == model]
        qualified_capture = any(row.get('status') == 'MEASURED_MATCHED_CAPTURE' for row in actual)
        manifest.append(dict(name=model + '_motivation',
            status=('QUALIFIED_SAME_RUN_CAPTURE_AVAILABLE' if qualified_capture else
                    'ONLY_REJECTED_OR_MISSING_CAPTURES') if actual else 'NOT_MEASURED',
            captures=actual,
            scope='New same-run kernel timelines require a qualified matched capture. Earlier motivation evidence remains in its original namespace.'))
        paired = []
        static = [row for row in measured if row['load'] == 'heavy' and not row['dynamic']]
        for loop in [row for row in static if row['arm'] == 'Loop_Static']:
            loop_path = ROOT / 'raw' / loop['name'] / 'split_work_and_payload.json'
            if not loop_path.exists():
                continue
            lw = json.loads(loop_path.read_text())
            target_rows = lw.get('measured_D_prompt_rows_from_active_counter', 0)
            if target_rows <= 0:
                continue
            for token in [row for row in static if row['arm'] == 'Token_Static']:
                token_path = ROOT / 'raw' / token['name'] / 'split_work_and_payload.json'
                if not token_path.exists():
                    continue
                tw = json.loads(token_path.read_text())
                if tw.get('measured_D_prompt_rows_from_active_counter') == target_rows:
                    paired.append((loop, token, lw, tw))
        if paired:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
            positions, labels = [], []
            selected = []
            for index, (loop, token, lw, tw) in enumerate(paired):
                for offset, row, work, color in ((-.18, loop, lw, '#3977a9'), (.18, token, tw, '#c86b35')):
                    x = index + offset
                    axes[0].bar(x, work['measured_D_prompt_rows_from_active_counter'] / row['count'], width=.34, color=color)
                    axes[1].bar(x, work['measured_transport_payload_bytes'] / row['count'] / 1024**2, width=.34, color=color)
                    value = row.get('D_prompt_device_enclosure_ms_p50')
                    if value is not None:
                        axes[2].bar(x, value, width=.34, color=color)
                    selected.append(row)
                positions.append(index)
                labels.append(f"Loop{loop['cut']} / Token{token['prefix']}")
            for ax, title, unit in zip(axes,
                    ('Active D recurrent rows', 'NIXL data payload per request', 'D prompt device enclosure p50'),
                    ('Rows per request', 'MiB per request', 'Milliseconds')):
                ax.set(title=title, ylabel=unit)
                ax.set_xticks(positions, labels, rotation=60, ha='right')
                ax.grid(axis='y', alpha=.2)
            fig.suptitle(f'{model.title()} | matched active rows, unequal total work | blue Loop, orange Token', y=1.02)
            save(fig, model + '_matched_rows_costs_not_a_novelty_claim', selected,
                'Active execution counters and measured NIXL data payloads. Device enclosures include scheduling gaps and interleaved work. Equal rows do not match causal Attention, Prelude, Coda, or padded work. This figure does not certify SLO-safe or Loop-exclusive advantage.')
        else:
            manifest.append(dict(name=model + '_matched_work_advantage', status='NOT_MEASURED',
                scope='Equal rows are not equal Attention work. No Loop-specific advantage is inferred from structural accounting.'))
    marker_free = ROOT / 'state/marker_free_serving_analysis.json'
    if marker_free.exists():
        controls = json.loads(marker_free.read_text())['results']
        for model in ('ouro', 'huginn'):
            selected = [row for row in controls if row['model'] == model and row['passed']]
            if not selected:
                continue
            fig, axes = plt.subplots(1, 4, figsize=(12, 4.))
            for ax, field, title in zip(axes,
                    ('TTFT_p95_ms', 'joint_SLO_goodput', 'TPOT_p95_ms', 'ITL_p99_ms'),
                    ('Request TTFT p95', 'Joint-SLO goodput', 'TPOT p95', 'Delivery ITL p99')):
                ax.bar(range(len(selected)), [row['metrics'][field] for row in selected], color='#3977a9')
                ax.set_xticks(range(len(selected)), [row['plan'] for row in selected], rotation=45, ha='right')
                ax.set(title=title, ylabel='Requests/s' if field == 'joint_SLO_goodput' else 'Milliseconds')
                ax.grid(axis='y', alpha=.2)
            frozen = selected[0]['metrics']['frozen_SLO']
            axes[2].axhline(.9 * frozen['TPOT_ms'], color='black', ls='--')
            axes[3].axhline(frozen['ITL_ms'], color='black', ls='--')
            fig.suptitle(f'{model.title()} | primary heavy-load cuts | marker-free N128 | one start\n'
                'Service TTFT is NOT MEASURED in these executions.', y=1.04)
            sources = [dict(source=f"raw/{row['name']}/marker_free_serving_analysis.json",
                source_SHA256=digest(ROOT / 'raw' / row['name'] / 'marker_free_serving_analysis.json')) for row in selected]
            save(fig, model + '_marker_free_primary_static_comparison', sources,
                'Execution-start markers and CUPTI capture are disabled. Same primary inputs and arrival trace. '
                'These are the primary cuts, not the best cut oracle. Service TTFT cannot be imported from another execution. '
                'The failed marker ON/OFF comparison remains failed.')
    write(ROOT / 'figure_manifest.json', dict(figures=manifest,
        scope='Actual request measurements with explicit instrumentation status. Marker-free controls remain separate. Missing GPU and novelty evidence remains explicit.'))


if __name__ == '__main__':
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    initialize()
    with quiet_analysis('serving_figures'):
        main()
