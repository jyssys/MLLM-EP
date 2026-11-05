"""Plot measured results and mark absent evidence without synthetic values."""
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

COLORS = {'native': '#444444', 'zero': '#999999', 'token': '#0072B2',
          'loop': '#D55E00', '2d': '#009E73'}


def family(method):
    return next((key for key in ('native', 'zero', 'token', 'loop', '2d')
                 if method.startswith(key)), 'native')


def load(name, default):
    path = ROOT / name
    return json.loads(path.read_text()) if path.exists() else default


def missing(ax, text='NOT_MEASURED'):
    ax.text(.5, .5, text, ha='center', va='center', transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def grouped(ax, rows, metric, title, unit, *, model=None):
    selected = [row for row in rows if (model is None or row['model'] == model)
                and isinstance(row.get(metric), (int, float))]
    ax.set_title(title)
    ax.set_ylabel(unit)
    if not selected:
        missing(ax)
        return
    keys = sorted({(row['model'], row['method']) for row in selected},
                  key=lambda key: (key[0], list(COLORS).index(family(key[1])), key[1]))
    for index, (name, method) in enumerate(keys):
        values = [row[metric] for row in selected if (row['model'], row['method']) == (name, method)]
        ax.bar(index, statistics.mean(values), color=COLORS[family(method)], alpha=.65)
        offsets = [0.] if len(values) == 1 else [(.32 * i / (len(values) - 1) - .16) for i in range(len(values))]
        ax.scatter([index + offset for offset in offsets], values, color='black', s=14, zorder=3)
    labels = []
    for name, method in keys:
        failed = any(row.get('Decode_SLO_screen') is False for row in selected
                     if (row['model'], row['method']) == (name, method))
        labels.append(name[:1].upper() + ':' + method + (' [screen x]' if failed else ''))
    ax.set_xticks(range(len(keys)), labels, rotation=60, ha='right')
    ax.grid(axis='y', alpha=.2)


def main():
    initialize()
    output = ROOT / 'figures'
    output.mkdir(exist_ok=True)
    manifest = []
    results = load('serving_results.json', [])
    rows = [row for row in results if row['correctness_passed'] and not row['calibration'] and not row.get('oracle_validation_only')]
    held = [row for row in rows if row['data_split'] == 'heldout' and row['load'] == 'medium']
    clean = [row for row in held if not row['markers']]
    accepted = {row['on'] for row in load('marker_interference.json', []) if row['global_wall_screen_passed']}
    observed = [row for row in held if row['markers'] and row['name'] in accepted]
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, stem, scope, status):
        fig.savefig(output / (stem + '.png'), dpi=180, bbox_inches='tight')
        fig.savefig(output / (stem + '.pdf'), bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(figure=stem, files=[f'figures/{stem}.png', f'figures/{stem}.pdf'],
            status=status, scope=scope))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    grouped(axes[0, 0], clean, 'TTFT_p95_ms', 'Held-out request TTFT p95 — marker-free', 'ms')
    grouped(axes[0, 1], observed, 'service_TTFT_ms_p95', 'Service TTFT p95 — accepted global marker screen', 'ms')
    grouped(axes[1, 0], clean, 'E2E_p95_ms', 'Held-out E2E p95 — marker-free', 'ms')
    for ax in (axes[0, 0], axes[1, 0]):
        ax.set_yscale('log')
        ax.set_ylabel('ms — logarithmic scale')
    graph = load('graph_cache_results.json', [])
    graph_held = [dict(row, hit_percent=100 * row['measured_existing_hit_ratio']) for row in graph
                  if '_heldout_' in row['name'] and row['role'] == 'prefill'
                  and not row['name'].startswith('v_')
                  and row.get('measured_existing_hit_ratio') is not None]
    grouped(axes[1, 1], graph_held, 'hit_percent', 'P graph hits among measured packed selections', '%')
    fig.suptitle('Native-path runtime validation\nLatency dots are per-run percentiles. Graph dots include separate ON/OFF audits. Service timing is instrumented.')
    save(fig, 'figure_1_native_path_runtime_validation',
         'Measured finite-cohort latency and native graph counters. No graph counter is kernel busy.',
         'MEASURED' if held else 'NOT_MEASURED')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    huginn = [row for row in clean if row['model'] == 'huginn' and not row['method'].startswith('2d_')]
    grouped(axes[0, 0], huginn, 'TTFT_p95_ms', 'Huginn held-out TTFT p95', 'ms')
    grouped(axes[0, 1], [row for row in observed if not row['method'].startswith('2d_')],
            'service_TTFT_ms_p95', 'Huginn instrumented Service TTFT p95', 'ms', model='huginn')
    pairs = load('paired_confirmation_ratios.json', {}).get('huginn', {}).get('marker_free', {})
    accepted_pairs = load('paired_marker_accepted_ratios.json', {}).get('huginn', {})
    methods = [method for method in pairs if method != 'native' and not method.startswith('2d_')]
    if methods:
        for index, method in enumerate(methods):
            entry = pairs[method].get('TTFT_p95_ms', {})
            value = entry.get('geometric_mean_ratio')
            if value is None:
                continue
            interval = entry.get('ratio_CI95')
            axes[1, 0].errorbar(index, value,
                yerr=[[value - interval[0]], [interval[1] - value]] if interval else None,
                fmt='o', color=COLORS[family(method)], capsize=4)
            axes[1, 0].annotate('n=' + str(entry['starts']), (index, value), xytext=(4, 6), textcoords='offset points')
            observed_entry = accepted_pairs.get(method, {}).get('TTFT_p95_ms', {})
            observed_value = observed_entry.get('geometric_mean_ratio')
            observed_interval = observed_entry.get('ratio_CI95')
            if observed_value is not None:
                axes[1, 0].errorbar(index + .2, observed_value,
                    yerr=[[observed_value - observed_interval[0]], [observed_interval[1] - observed_value]]
                         if observed_interval else None,
                    fmt='s', markerfacecolor='none', color=COLORS[family(method)], capsize=3)
        axes[1, 0].axhline(1, color='black', linestyle='--')
        axes[1, 0].set_xticks(range(len(methods)), methods, rotation=35, ha='right')
        axes[1, 0].set_ylabel('Candidate / Native TTFT ratio, paired 95% CI\nCircle: marker-free. Open square: accepted ON.')
    else:
        missing(axes[1, 0])
    grouped(axes[1, 1], huginn, 'ITL_p99_ms', 'Huginn Decode tail — marker-free', 'ms')
    slo = load('configs/frozen_SLO.json', {}).get('huginn_L2048', {})
    # The frozen schema is retained. Locate the exact existing ITL field.
    itl = next((v for k, v in slo.items() if 'itl' in k.lower() and isinstance(v, (int, float))), None)
    if itl is not None:
        axes[1, 1].axhline(itl, color='red', linestyle='--', label='Frozen ITL gate')
        axes[1, 1].legend()
    fig.suptitle('Huginn Loop31 confirmation\nHeld-out prompts and seeds did not select the cut. All 2D results remain in Figures 1 and 4.')
    save(fig, 'figure_2_huginn_loop31_heldout_confirmation',
         'Measured paired starts. CI uses start-level log ratios, not independent-token resampling. Marker ON/OFF evidence remains separate.',
         'MEASURED' if huginn else 'NOT_MEASURED')

    controls = [row for row in rows if row['name'].startswith('h_ouro_')]
    discovery = [row for row in rows if row['model'] == 'ouro' and row['method'] == '2d_c3_k3072'
                 and row['data_split'] == 'discovery' and row['markers']]
    sensitivity = controls + discovery
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, metric, label in zip(axes, ['service_TTFT_ms_p95', 'TTFT_p95_ms', 'ITL_p99_ms'],
                                ['Service TTFT p95 — instrumented', 'Request TTFT p95 — instrumented', 'Decode ITL p99 — instrumented']):
        ax.set_title(label)
        if not sensitivity:
            missing(ax)
            continue
        for index, row in enumerate(sensitivity):
            ax.scatter(index, row.get(metric, math.nan), color='#009E73' if row['cache_policy'] == 'primary_reserved' else '#CC79A7')
        ax.set_xticks(range(len(sensitivity)), [row['cache_policy'] + str(row['graph_budget']) + '\ns' + str(row['start']) for row in sensitivity], rotation=50, ha='right')
        ax.set_ylabel('ms')
    sensitivity_status = 'MEASURED' if controls else 'NOT_MEASURED'
    sensitivity_scope = 'Measured control package. Cache policy and serial primary warm-up change together. Do not attribute the difference to a single sub-intervention.'
    if not controls:
        for ax in axes:
            ax.clear()
        current = [row for row in graph if row['name'].startswith('c_ouro_2d_c3_k3072_discovery_')
                   and row['role'] == 'prefill']
        keys = sorted({tuple(entry['key']) for row in current for entry in row['entries']})
        if current and keys:
            matrix = [[int(any(tuple(entry['key']) == key and entry['captured']
                               for entry in row['entries'])) for row in current] for key in keys]
            axes[0].imshow(matrix, vmin=0, vmax=1, cmap='Greens', aspect='auto')
            axes[0].set_xticks(range(len(current)), ['s' + row['name'].split('_s')[-1].split('_')[0] for row in current])
            axes[0].set_yticks(range(len(keys)), [f'rows {key[0]}, q {key[2]}' for key in keys])
            axes[0].set_xticks([value - .5 for value in range(len(current) + 1)], minor=True)
            axes[0].set_yticks([value - .5 for value in range(len(keys) + 1)], minor=True)
            axes[0].grid(which='minor', color='white', linewidth=1)
            axes[0].tick_params(which='minor', bottom=False, left=False)
            for y, row in enumerate(matrix):
                for x, captured in enumerate(row):
                    axes[0].text(x, y, 'yes' if captured else 'no', ha='center', va='center',
                                 color='white' if captured else 'black', fontsize=8)
            axes[0].set_xlabel('Independent discovery start')
            axes[0].set_title('Captured exact P query geometries')
            for index, row in enumerate(current):
                axes[1].bar(index, 100 * row['measured_existing_hit_ratio'], color='#009E73')
                axes[1].annotate(f'cold {row["cold_entries"]}\neager {row["fallbacks"]}',
                    (index, 97), ha='center', va='top', fontsize=8)
            axes[1].set_xticks(range(len(current)), ['s' + row['name'].split('_s')[-1].split('_')[0] for row in current])
            axes[1].set_ylim(0, 105)
            axes[1].set_ylabel('Existing-entry hit ratio, %')
            axes[1].set_title('Measured packed-P selections')
            sensitivity_status = 'MEASURED_CACHE_COVERAGE_LEGACY_LATENCY_CONTROL_NOT_RUN'
        else:
            missing(axes[0])
            missing(axes[1])
        grouped(axes[2], [row for row in observed if row['model'] == 'ouro'
                         and row['method'] in ('native', '2d_c3_k3072')],
                'service_TTFT_ms_p95', 'Current eight-entry Service TTFT', 'ms')
        sensitivity_scope = ('Captured geometry and hit counters use the three discovery starts. '
            'Current Service TTFT uses accepted held-out marker runs. These cohorts are not a paired before/after latency control. '
            'Legacy-budget sensitivity remains NOT_RUN_TIME_BUDGET.')
        fig.suptitle('Ouro current-cache coverage and residual runtime cost\nLegacy-budget latency controls were not completed. Cache counters are not kernel busy.')
    else:
        fig.suptitle('Ouro graph-cache sensitivity\nLegacy4 versus reserved4/reserved8. Main comparisons share the eight-entry policy.')
    save(fig, 'figure_3_ouro_graph_cache_sensitivity',
         sensitivity_scope, sensitivity_status)

    work = {row['name']: row for row in load('work_and_payload.json', [])}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for index, model in enumerate(('ouro', 'huginn')):
        chosen = [row for row in clean if row['model'] == model]
        for row in chosen:
            record = work.get(row['name'])
            if record is None:
                continue
            d_rows = record['D_prompt_recurrent_rows'] / row['count']
            color = COLORS[family(row['method'])]
            marker = 'o' if row['Decode_SLO_screen'] else 'x'
            axes[index, 0].scatter(d_rows, row['TTFT_p95_ms'], color=color, marker=marker)
            payload = record['counters']['prefill']['transport_bytes_sent'] / row['count'] / 1024**2
            axes[index, 1].scatter(payload, row['joint_SLO_goodput'], color=color, marker=marker)
        for ax in axes[index]:
            if not chosen:
                missing(ax)
            ax.set_title(model.capitalize() + ' — held-out marker-free')
        axes[index, 0].set_xlabel('Actual D prompt recurrent rows / request — not FLOPs')
        axes[index, 0].set_ylabel('Request TTFT p95, ms')
        axes[index, 1].set_xlabel('Measured P payload counter / request, MiB')
        axes[index, 1].set_ylabel('Joint SLO-goodput, requests/s')
    handles = [plt.Line2D([], [], marker='o', linestyle='', color=color, label=label) for label, color in COLORS.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=5)
    fig.suptitle('Token versus Loop versus 2D\nCrosses fail the additional-work Decode screen.\nEqual recurrent rows do not mean equal Attention or padded GEMM work.')
    save(fig, 'figure_4_token_loop_2d_work_and_payload',
         'Measured semantic rows and transport payload counters. Control messages and network headers are excluded. No equal-FLOP claim.',
         'MEASURED' if clean else 'NOT_MEASURED')

    validation = load('oracle_static_validation.json', {})
    headroom = load('oracle_headroom.json', {})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    predictions = validation.get('rows', [])
    if predictions:
        markers = {'TTFT_p95_ms': 'o', 'TPOT_p95_ms': '^', 'ITL_p99_ms': 'x', 'E2E_p95_ms': 's'}
        plotted = []
        for model in ('ouro', 'huginn'):
            for metric, marker in markers.items():
                sample = [row for row in predictions if row['model'] == model and not row['markers']
                          and row['metric'] == metric and row['measured'] > 0 and row['predicted'] > 0]
                plotted += sample
                if sample:
                    axes[0].scatter([row['measured'] for row in sample], [row['predicted'] for row in sample],
                        color='#0072B2' if model == 'ouro' else '#D55E00', marker=marker,
                        label=model + ' ' + metric.split('_')[0], alpha=.65, s=24)
        if plotted:
            lower = min(min(row['measured'], row['predicted']) for row in plotted) * .8
            upper = max(max(row['measured'], row['predicted']) for row in plotted) * 1.2
            axes[0].plot([lower, upper], [lower, upper], '--', color='black')
            axes[0].set_xscale('log')
            axes[0].set_yscale('log')
            axes[0].legend(fontsize=7, ncol=2)
    else:
        missing(axes[0], 'NOT_MEASURED\nIndependent static-model validation is incomplete.')
    axes[0].set_xlabel('Measured held-out latency, ms')
    axes[0].set_ylabel('Frozen model prediction, ms')
    axes[0].set_title('Offline model validation — not measured speedup')
    modeled = headroom.get('results', [])
    if modeled:
        for index, entry in enumerate(modeled):
            for offset, key in enumerate(('token', 'loop', 'two_dimensional')):
                axes[1].bar(index * 4 + offset, entry[key]['metrics']['joint_SLO_goodput'],
                            color=COLORS[('token', 'loop', '2d')[offset]])
        axes[1].set_ylabel('MODELED joint SLO-goodput, requests/s')
        axes[1].set_xticks(range(8), ['O Token', 'O Loop', 'O 2D', '', 'H Token', 'H Loop', 'H 2D', ''], rotation=45, ha='right')
    else:
        missing(axes[1], headroom.get('status', 'NOT_MEASURED') + '\nFeasible-oracle headroom is not established.')
    axes[1].set_title('Restricted offline candidates — not an optimality bound')
    fig.suptitle('Oracle predictions remain separate from measured serving')
    save(fig, 'figure_5_validated_oracle_headroom',
         'MODELED candidate search only after independent error screens. Inaccurate or missing calibration blocks headroom claims. Mixed-frontier plans are not qualified online execution.',
         headroom.get('status', 'NOT_MEASURED'))
    write(ROOT / 'figure_manifest.json', dict(time=time.time(), figures=manifest,
        scope='Measured results, structural accounting, and modeled predictions remain distinct. No image contains invented kernel activity or idle intervals.'))


if __name__ == '__main__':
    main()
