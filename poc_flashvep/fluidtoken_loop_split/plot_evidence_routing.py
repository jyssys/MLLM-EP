"""Plot retained host pauses and every eligible matched-geometry clean repeat."""
import json
from .common import *


def main():
    configure()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .static_screen import metrics
    before = ['recovery_huginn_slots4_graph4_tail512',
              'recovery_huginn_slots4_graph4_joint_s1_clean512',
              'recovery_huginn_slots4_graph4_Ponly_joint_s1_clean512']
    after = [f'recovery_huginn_async_evidence_slots4_graph4_joint_s{i}_clean512' for i in (1, 2, 3)]
    rows = []
    for group, names in [('Original logging', before), ('Async campaign logging', after)]:
        for name in names:
            result = ROOT / 'raw' / name / 'dispatcher_result.json'
            if not result.exists() or not json.loads(result.read_text()).get('complete'):
                continue
            cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
            assert cfg['D_slots'] == cfg['graph_rows'] == 4
            assert not cfg.get('kernel_activity_seconds')
            rows.append(dict(group=group, name=name, seed=cfg['arrival_seed'], **metrics(name)))
    diagnostic = ROOT / 'raw/recovery_huginn_graph4_evidence_io_diagnostic512'
    events = []
    for path in diagnostic.glob('host_pauses_*.json'):
        value = json.loads(path.read_text())
        events += [dict(row, pid=value['pid']) for row in value['events']
                   if row['method'] == 'evidence_append' and 8416533.4 <= row['start'] <= 8416533.5]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    origins = {'1713311': 'P evidence callback', '1713312': 'D evidence callback', '1713290': 'Coordinator callback'}
    for i, event in enumerate(sorted(events, key=lambda row: row['pid'])):
        axes[0].broken_barh([((event['start'] - 8416533.4) * 1000, event['seconds'] * 1000)],
                            (i - .3, .6), facecolors='#b34b39')
        axes[0].text((event['start'] - 8416533.4) * 1000 + 5, i,
                     f"{event['seconds'] * 1000:.1f} ms", va='center', color='white')
    axes[0].set_yticks(range(len(events)), [origins[str(row['pid'])] for row in sorted(events, key=lambda row: row['pid'])])
    axes[0].set(xlabel='Host time from diagnostic anchor (ms)', title='Measured foreground append pauses')
    colors = {'Original logging': '#b34b39', 'Async campaign logging': '#287f8e'}
    for axis, field, label in [(axes[1], 'ITL_p99_ms', 'ITL p99 (ms)'),
                               (axes[2], 'maximum_ITL_ms', 'Maximum ITL (ms)')]:
        for i, row in enumerate(rows):
            axis.scatter(i, row[field], color=colors[row['group']], s=55)
            axis.annotate(f"{row[field]:.1f}", (i, row[field]), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
        axis.set_xticks(range(len(rows)), [('Before' if r['group'] == 'Original logging' else 'After') + '\n' + str(r['seed'])[-4:] for r in rows])
        axis.set(ylabel=label, xlabel='Independent clean run / arrival seed', title='All retained geometry-4 controls')
        axis.grid(axis='y', alpha=.25)
        axis.margins(y=.15)
    axes[1].axhline(192.821253818, color='black', linestyle='--', label='Frozen p99 limit')
    axes[1].legend(fontsize=8)
    fig.suptitle('Generic logging-path control — not a Token/Loop Split speedup')
    prefix = ROOT / 'figures/evidence_routing_control'
    fig.savefig(prefix.with_suffix('.png'), dpi=180)
    fig.savefig(prefix.with_suffix('.svg'))
    plt.close(fig)
    write(prefix.with_suffix('.json'), dict(rows=rows, diagnostic_events=events,
        scope='Diagnostic host pauses and separate clean-run latency. No addition of overlapping intervals. No claim that all maximum outliers disappear.'))


if __name__ == '__main__':
    main()
