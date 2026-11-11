"""Plot measured selection controls. Never replace missing experiments with zero."""
import argparse
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirmation-tag')
    args = parser.parse_args()
    initialize()
    folder = ROOT/'figures'
    folder.mkdir(exist_ok=True)
    manifest = []
    view = analysis()

    def measured(name):
        path = ROOT/'raw'/name/'dispatcher_result.json'
        if not path.exists() or not json.loads(path.read_text()).get('complete'):
            return None
        result = view.metrics(name)
        result['source'] = str(ROOT/'raw'/name)
        return result

    def save(fig, name, sources, scope):
        fig.tight_layout()
        for extension in ('png', 'svg'):
            fig.savefig(folder/(name+'.'+extension), dpi=160, bbox_inches='tight')
        plt.close(fig)
        manifest.append(dict(name=name, files=[f'figures/{name}.{s}' for s in ('png', 'svg')],
            sources=sources, scope=scope))

    names = [
        ('Zero / original runtime', 'selection_zero_c_seed2026110940'),
        ('Zero / fixed R32 submission', 'selection_fastzero_seed2026110940'),
        ('Selective / no migration', 'selection_matchedwire_nomigration_seed2026110940'),
        ('All checkpoints / no migration', 'lifetime3_selection_matchedwire_checkpoint_seed2026110940'),
        ('Zero / deferred CODA v2', 'deferred2_zero_seed2026110940'),
        ('Static Loop31 / deferred CODA v2', 'deferred2_loop31_seed2026110940')]
    values = [(label, measured(name)) for label, name in names]
    values = [(label, value) for label, value in values if value]
    if values:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        labels = [label for label, _ in values]
        for ax, key, title, scale in zip(axes, ('TTFT_p95_ms', 'ITL_p99_ms'),
                ('Request TTFT p95 (s)', 'Decode ITL p99 (ms)'), (1000, 1)):
            ax.barh(labels, [value[key]/scale for _, value in values], color='#447e9f')
            ax.set_xlabel(title)
            ax.grid(axis='x', alpha=.2)
        fig.suptitle('Huginn R32: measured N128 selection controls, one paired Arrival seed')
        save(fig, 'figure_1_runtime_and_checkpoint_controls',
            [v['name'] for _, v in values],
            'Marker-free selection observations, not held-out confidence intervals. Request TTFT includes Queue waiting. Service TTFT is NOT_MEASURED. The original-runtime row is an independent start.')

    values = []
    for label, family in (('Normal context', 'deferred2'), ('16 SM', 'lifetime3_spatial_sm16'),
            ('32 SM', 'lifetime3_spatial_sm32')):
        baseline = measured(f'{family}_zero_seed2026110940')
        for arm in ('loop31', 'token1984'):
            value = measured(f'{family}_{arm}_seed2026110940')
            if baseline and value:
                values.append(dict(label=f'{label}\n{arm}', name=value['name'],
                    baseline=baseline['name'],
                    TTFT=value['TTFT_p95_ms']/baseline['TTFT_p95_ms'],
                    ITL=value['ITL_p99_ms']/baseline['ITL_p99_ms'],
                    TPOT=value['TPOT_p95_ms']/baseline['TPOT_p95_ms']))
    if values:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = list(range(len(values)))
        for offset, key, color in ((-.25, 'TTFT', '#447e9f'), (0, 'ITL', '#c65c52'), (.25, 'TPOT', '#d1a34c')):
            ax.bar([v+offset for v in x], [r[key] for r in values], .24, label=key, color=color)
        ax.axhline(1., color='black', linewidth=.8)
        ax.axhline(1.05, color='#c65c52', linestyle='--', label='Decode design target')
        ax.set_xticks(x, [r['label'] for r in values])
        ax.set_ylabel('Ratio to the matched Native-path Zero start')
        ax.set_title('Spatial restriction alone does not protect Native-relative Decode tails')
        ax.legend(ncol=4, fontsize=8)
        save(fig, 'figure_2_spatial_quantum_tradeoff',
            [r[k] for r in values for k in ('name', 'baseline')],
            'Measured static full-prompt Loop31 and 64-token suffix selections. Normal and Green contexts have different execution resources. Each method has its own matched Zero. No hardware concurrency claim applies.')
        write(folder/'spatial_selection_ratios.json', values)

    paired_path = (ROOT/'paired_host_attribution'/
        'deferred2_loop31_seed2026110940_vs_deferred2_zero_seed2026110940.json')
    if paired_path.exists():
        observation = json.loads(paired_path.read_text())
        delta = observation['candidate_minus_reference_ms']
        fields = [
            ('Arrival → admission', 'arrival_to_admit'),
            ('Admission → P start receipt', 'admit_to_P_start_receipt'),
            ('P start → release receipt', 'P_start_to_release_receipt'),
            ('P release → wire commit receipt', 'P_release_to_wire_commit_receipt'),
            ('Wire commit → D receipt', 'wire_commit_to_D_received_receipt'),
            ('D receipt → first output', 'D_received_to_output_delivery'),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        ax, tail = axes
        means = [delta[key]['mean'] for _,key in fields]
        ax.barh([label for label,_ in fields], means,
                color=['#447e9f' if value < 0 else '#c65c52' for value in means])
        ax.invert_yaxis()
        ax.set_xlabel('Mean Loop31 minus Zero (ms, signed log scale)')
        ax.set_xscale('symlog', linthresh=10)
        ax.axvline(0., color='black', linewidth=.8)
        ax.set_title(f'{observation["paired_receipts"]} paired Host partitions')
        live = observation['delivered_ITL_by_live_Decode_at_previous_output']
        for offset, key, label, color in ((-.15, 'reference', 'Zero', '#447e9f'),
                                          (.15, 'candidate', 'Loop31', '#c65c52')):
            points = [(int(batch), vals) for batch,vals in live[observation[key]].items()]
            tail.bar([batch+offset for batch,_ in points],
                     [vals['p99'] for _,vals in points], .28, label=label, color=color)
            for batch,vals in points:
                tail.text(batch+offset, vals['p99']+1., str(vals['n']), ha='center', fontsize=8)
        tail.set_xticks(sorted({int(n) for value in live.values() for n in value}))
        tail.set_xlabel('Live Decode requests at preceding output')
        tail.set_ylabel('Delivered ITL p99 (ms); numbers show interval counts')
        tail.margins(y=.15)
        tail.legend()
        tail.set_title('Descriptive load strata, not a physical batch size')
        fig.suptitle('Full-device Loop31: less P queueing, greater Decode tail')
        save(fig, 'figure_4_paired_queue_and_decode_tail',
            [observation['reference'], observation['candidate'], str(paired_path)],
            'Measured signed Host intervals on one paired Arrival trace. Means share identical requests. Not a GPU critical-path or DMA decomposition. The Decode strata are descriptive, not causal.')

    reference = measured('deferred2_zero_seed2026110940')
    if reference:
        probes = [
            ('Zero', reference['name'], '#447e9f', 'o'),
            ('All checkpoints', 'lifetime3_selection_matchedwire_checkpoint_seed2026110940', '#777777', 's'),
            ('Previous Loop', 'lifetime3_selection_matched_oldloop_seed2026110940', '#777777', '^'),
            ('Adaptive forecast', 'forecast8b_output16_loop_seed2026110940', '#d1a34c', 'D'),
            ('Full-device Loop31', 'deferred2_loop31_seed2026110940', '#148979', 'o'),
            ('Priority Loop31', 'priority4_normal_loop31_seed2026110940', '#148979', '^'),
            ('Loop31 / 128 rows', 'quantum6_loop31_128_seed2026110940', '#148979', 's'),
            ('Loop31 / 256 rows', 'quantum6_loop31_256_seed2026110940', '#148979', 'v'),
            ('Loop31 / 512 rows', 'quantum6_loop31_512_seed2026110940', '#148979', 'P'),
            ('Uncapped Token64', 'quantum6_token1984_1_seed2026110940', '#c65c52', 's'),
            ('Corrected Token64', 'token16_prefix1984_cap8_seed2026110940', '#c65c52', 'v'),
            ('SM32 Token64', 'lifetime3_spatial_sm32_token1984_seed2026110940', '#c65c52', 'o'),
            ('Corrected Token256', 'token16_prefix1792_cap8_seed2026110940', '#c65c52', '^'),
            ('Full-prompt Loop28', 'match18_loop28_full_seed2026110940', '#6e5c9b', 'o'),
            ('Loop28 / 256 rows', 'match18_loop28_tile256_seed2026110940', '#6e5c9b', 's'),
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        names = []
        for label,name,color,marker in probes:
            value = measured(name)
            if not value:
                continue
            names.append(name)
            ax.scatter(value['TTFT_p95_ms']/1000, value['ITL_p99_ms'], c=color,
                marker=marker, s=45+value['offloaded_requests'], label=label)
        ax.axvline(.95*reference['TTFT_p95_ms']/1000, color='#447e9f', linestyle='--')
        ax.axhline(1.05*reference['ITL_p99_ms'], color='#447e9f', linestyle='--')
        ax.axhline(192.8212538175284, color='#c65c52', linestyle=':', label='Absolute ITL p99 limit')
        ax.set_xlabel('Request TTFT p95 (s); lower is better')
        ax.set_ylabel('Decode ITL p99 (ms); lower is better')
        ax.set_title('Huginn 2048/16: one selection start per tested configuration')
        ax.legend(loc='upper right', ncol=2, fontsize=8)
        save(fig, 'figure_5_selection_ttft_decode_tradeoff', names,
            'Observed selection runs. Dashed lines are relative to full-device Zero. Marker area increases with actual offloaded request count. Different execution policies require their own matched controls. This plot cannot establish GO.')

    if args.confirmation_tag:
        tag = args.confirmation_tag
        path = ROOT/'confirmation'/(tag+'_paired_start_ratios.json')
        if path.exists():
            ratios = json.loads(path.read_text())
            subset = [r for r in ratios if r['reference'] == 'zero'
                and r['arm'] != 'zerorepeat'
                and r['metric'] in ('TTFT_p95_ms', 'E2E_p95_ms', 'ITL_p99_ms', 'TPOT_p95_ms')
                and r['geometric_ratio'] is not None]
            if subset:
                labels = dict(loop='Loop31 / 512-token tiles', loopfour='Loop28 / full prompt',
                    token='Token256 / eight-Loop groups', tokensafe='Token64 / SM32 / uncapped',
                    adaptive='Adaptive Loop / early KV', lateloop='Adaptive Loop / late KV',
                    dyntoken='Adaptive Token')
                fig, ax = plt.subplots(figsize=(10, max(4, len(subset)*.32)))
                for index, row in enumerate(subset):
                    point = row['geometric_ratio']
                    interval = row['bootstrap_95_CI']
                    ax.plot(point, index, 'o', color='#447e9f')
                    if interval:
                        ax.plot(interval, [index, index], color='#447e9f')
                ax.set_yticks(range(len(subset)), [labels.get(r['arm'], r['arm'])+' / '+r['metric'] for r in subset])
                ax.axvline(1., color='black', linewidth=.8)
                ax.axvline(1.05, color='#c65c52', linestyle='--')
                ax.set_xlabel('Paired geometric ratio with start-level bootstrap 95% CI')
                ax.set_title('Held-out Native-relative confirmation')
                save(fig, 'figure_3_held_out_confirmation', [str(path)],
                    'Paired independent starts by Arrival seed. The repeated Zero is a sensitivity control. Three seeds provide limited confidence coverage. Ratios below one improve latency.')
    write(ROOT/'figure_manifest.json', manifest)
    print(json.dumps(dict(figures=len(manifest), directory=str(folder))))


if __name__ == '__main__':
    main()
