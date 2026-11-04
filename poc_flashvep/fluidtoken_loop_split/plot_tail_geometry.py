"""Plot matched clean controls without claiming independent confirmation."""
from .common import *


def main():
    configure()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('tail_geometry_figure'):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        state = json.loads((ROOT / 'state/tail_geometry_control.json').read_text())
        assert state['status'] == 'GEOMETRY_SCREEN_COMPLETE'
        rows = [r for r in state['results'] if r.get('complete')]
        labels = [f"D slots {r['D_slots']}\nGraph rows {r['graph_rows']}" for r in rows]
        fallbacks = []
        for row in rows:
            samples = [json.loads(p.read_text()) for p in (ROOT / 'raw' / row['name']).glob('native_graph_coverage_*.json')]
            fallbacks.append(next(r['measured_delta']['fallbacks'] for r in samples if r['role'] == 'decode'))
        values = [[r['metrics']['ITL_p99_ms'] for r in rows],
                  [r['metrics']['TPOT_p95_ms'] for r in rows],
                  [r['metrics']['maximum_ITL_ms'] for r in rows], fallbacks]
        titles = ['Token ITL p99 (ms)', 'Request TPOT p95 (ms)', 'Maximum token ITL (ms)', 'Eager graph fallbacks (count)']
        slo = rows[0]['metrics']['frozen_SLO']
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for index, axis in enumerate(axes.flat):
            bars = axis.bar(labels, values[index], color=['#b34e42', '#367d65', '#4b71a5'])
            axis.bar_label(bars, fmt='%.2f' if index < 3 else '%d', padding=3)
            axis.set_title(titles[index])
            axis.grid(axis='y', alpha=.2)
            axis.set_ylim(0, max(values[index]) * 1.25 if max(values[index]) else 1)
            if index < 2:
                limit = slo['ITL_ms'] if index == 0 else .9 * slo['TPOT_ms']
                axis.axhline(limit, color='black', linestyle='--', label=f'Frozen gate: {limit:.2f} ms')
                axis.legend(fontsize=8)
        fig.suptitle('Huginn R32: matched clean Decode geometry controls\n'
            '2048/16, 512 requests, same arrival seed. One start per configuration. Not final recovery qualification.')
        output = ROOT / 'figures/decode_tail_geometry_controls'
        for extension in ('png', 'svg'):
            fig.savefig(str(output) + '.' + extension, dpi=160)
        plt.close(fig)
        write(ROOT / 'figures/decode_tail_geometry_manifest.json', dict(
            evidence='MEASURED_CLEAN_HOST_LATENCIES_AND_NATIVE_GRAPH_COUNTERS',
            runs=[r['name'] for r in rows], requests_per_run=512,
            independent_repeats_per_configuration=1,
            scope='Matched arrival trace. Slot changes also change batch composition. No confidence interval or same-run P/D headroom claim.'))


if __name__ == '__main__':
    main()
