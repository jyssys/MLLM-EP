"""Plot only completed exact static controls. Missing methods remain explicit."""
from .common import *


def main():
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('method_screen_figures'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        state_path = ROOT / 'state/static_screen.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        records = []
        labels = ('atomic_native', 'staged_native', 'staged_snapshot_full', 'token_partial', 'loop_partial')
        fields = ('TTFT_p95_ms', 'TPOT_p95_ms', 'ITL_p99_ms', 'E2E_p95_ms',
                  'requests_per_second', 'joint_SLO_goodput')
        titles = ('TTFT p95 (ms)', 'TPOT p95 (ms)', 'ITL p99 (ms)', 'E2E p95 (ms)',
                  'Completed requests/s', 'Joint-SLO goodput (requests/s)')
        for model in ('ouro', 'huginn'):
            rows = [row for row in state.get('results', [])
                    if row.get('model') == model and row.get('correct')]
            if not rows:
                records.append(dict(model=model, source_runs=[], evidence='NOT_MEASURED',
                    reason='No exact completed static screen exists. No placeholder performance figure was generated.'))
                continue
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            sources = []
            for axis, field, title in zip(axes.flat, fields, titles):
                for index, label in enumerate(labels):
                    found = [row for row in rows if label in row['name']]
                    if found:
                        row = found[-1]
                        value = row['metrics'][field]
                        axis.bar(index, value, color='#2456a5')
                        axis.annotate(f'{value:.3f}', (index, value), ha='center', va='bottom', fontsize=8)
                        if row['name'] not in sources:
                            sources.append(row['name'])
                    else:
                        axis.text(index, .05, 'NOT\nMEASURED', ha='center', va='bottom', fontsize=7,
                                  transform=axis.get_xaxis_transform())
                axis.set_title(title)
                axis.set_xticks(range(len(labels)), [s.replace('_', '\n') for s in labels], fontsize=8)
                axis.set_xlim(-.6, len(labels) - .4)
                axis.grid(axis='y', alpha=.2)
            fig.suptitle(f'{model}: N32 exploratory static controls\n'
                         'No dynamic result or stable-capacity claim. Scheduling and transport controls remain separate.')
            target = ROOT / 'figures' / f'{model}_static_method_screen'
            for extension in ('png', 'svg'):
                fig.savefig(str(target) + '.' + extension, dpi=160)
            plt.close(fig)
            records.append(dict(model=model, source_runs=sources,
                figure=str(target), evidence='MEASURED_N32_SCREEN' if sources else 'NOT_MEASURED',
                dynamic_token='NOT_MEASURED', dynamic_loop='NOT_MEASURED',
                limitation='One exploratory start per available control. No confidence interval or replicated method advantage.'))
        write(ROOT / 'figures/static_method_screen_manifest.json', records)


if __name__ == '__main__':
    main()
