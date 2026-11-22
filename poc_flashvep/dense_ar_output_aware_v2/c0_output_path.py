"""Locate the controlled-output axis contrast in additive host boundaries."""
import json
import time
import numpy as np
from .common import ROOT, write, digest
from .path_analysis import STAGES
from .statistics import paired_difference
from .c0_analysis import csv_save, INPUTS


GROUPS = (
    ('P queue and host admission', STAGES[:3]),
    ('P prepare to ready host enclosure', STAGES[3:4]),
    ('Ready to D submit', STAGES[4:5]),
    ('D admission and prepare', STAGES[5:7]),
    ('D prepare to first worker host enclosure', STAGES[7:8]),
    ('Worker to client', STAGES[8:9]),
)


def main():
    path = ROOT / 'analysis/c0_output_matrix_v2_host_path_v2.json'
    source = json.loads(path.read_text())
    indexed = {(r['input_bucket'], r['cut'], r['output'], r['start'], r['label']): r
               for r in source['starts'] if r['split'] == 'heldout'}
    results, plotted = [], []
    for bucket in range(4):
        for cut in (2, 4, 8, 12):
            endpoints = {}
            for output in (8, 192):
                endpoints[output] = {}
                for label, fields in GROUPS + (('Mean TTFT', ('TTFT_ms',)),
                                              ('Decode residency', ('D_decode_residency_ms',))):
                    layer = [sum(indexed[bucket, cut, output, start, 'Layer']['means'][field]
                                 for field in fields) for start in range(3)]
                    token = [sum(indexed[bucket, cut, output, start, 'Token']['means'][field]
                                 for field in fields) for start in range(3)]
                    endpoints[output][label] = dict(layer=layer, token=token,
                                                   delta=paired_difference(layer, token))
                    plotted.append(dict(input_bucket=bucket, input_label=INPUTS[bucket], cut=cut,
                                        output=output, host_group=label,
                                        Layer_start_mean_ms=float(np.mean(layer)),
                                        Token_start_mean_ms=float(np.mean(token)),
                                        Layer_minus_Token_ms=endpoints[output][label]['delta'],
                                        evidence='TRACE_ANALYSIS'))
            contrasts = {}
            for label in endpoints[8]:
                high = endpoints[192][label]
                low = endpoints[8][label]
                a = np.asarray(high['layer']) - np.asarray(high['token'])
                b = np.asarray(low['layer']) - np.asarray(low['token'])
                contrasts[label] = paired_difference(a, b)
            total = sum(contrasts[label]['estimate'] for label, _ in GROUPS)
            assert abs(total - contrasts['Mean TTFT']['estimate']) < 1e-3
            results.append(dict(input_bucket=bucket, cut=cut, paired_starts=3,
                                output192_minus8_in_Layer_minus_Token_mean_ms=contrasts,
                                endpoints=endpoints, evidence='TRACE_ANALYSIS'))
    write(ROOT / 'analysis/c0_output_host_path_contrast_v2.json', dict(
        time=time.time(), source_SHA256=digest(path), rows=results,
        scope='The original prompt IDs,Input,Cut,QPS and arrival seed match across each output endpoint. '
              'The stages sum to mean client TTFT,not TTFT p95. Prepare is a host boundary,not a GPU start. '
              'Output is a controlled intervention. The stage contrast locates its total axis effect but does not identify a causal mediator. '
              'Decode residency,source credit feedback,queue and host service can change together. '
              'C0 Mode0 Token is not the compiled C3 Stock Token. Intervals are paired start-level,pointwise and unadjusted.'))
    csv_save(ROOT / 'figures/c0_output_host_path_endpoint_values_v2.csv', plotted)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = plt.get_cmap('tab10').colors
    for bucket in range(4):
        fig, axes = plt.subplots(1, 4, figsize=(18, 5.8), sharey=True)
        records = []
        for ax, cut in zip(axes, (2, 4, 8, 12)):
            row = next(r for r in results if r['input_bucket'] == bucket and r['cut'] == cut)
            positive = np.zeros(2)
            negative = np.zeros(2)
            for index, (label, _) in enumerate(GROUPS):
                values = np.asarray([row['endpoints'][o][label]['delta']['estimate'] for o in (8, 192)])
                base = np.where(values >= 0, positive, negative)
                ax.bar([0, 1], values, bottom=base, color=colors[index], label=label)
                positive += np.maximum(values, 0)
                negative += np.minimum(values, 0)
                for output, value in zip((8, 192), values):
                    records.append(dict(input_bucket=bucket, cut=cut, output=output, host_group=label,
                                        Layer_minus_Token_mean_ms=float(value),
                                        CI95=row['endpoints'][output][label]['delta']['CI95'], starts=3,
                                        evidence='TRACE_ANALYSIS'))
            ax.axhline(0, color='black', linewidth=.7)
            ax.set(title=f'Last{cut}', xticks=[0, 1], xticklabels=['Output8', 'Output192'],
                   xlabel='Controlled output tokens')
            ax.grid(axis='y', alpha=.2)
        axes[0].set_ylabel('Layer minus Token mean host-stage latency (ms)')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(.5, .065))
        fig.suptitle(f'C0 actual batch1/1/1 | {INPUTS[bucket]} | paired N3')
        fig.text(.02, .015, 'TRACE_ANALYSIS. Signed stages sum to mean TTFT,not TTFT p95. Prepare is not a GPU kernel boundary.\n'
                 'Output changes residency,queue and capacity-relative load. This is not an isolated mediator intervention or C3 serving result.', fontsize=8)
        fig.tight_layout(rect=(0, .18, 1, .94))
        stem = f'c0_output_host_path_input{bucket}_v2'
        for extension in ('png', 'pdf'):
            fig.savefig(ROOT / 'figures' / f'{stem}.{extension}', dpi=170)
        csv_save(ROOT / 'figures' / f'{stem}.csv', records)
        plt.close(fig)
    print('C0_CONTROLLED_OUTPUT_HOST_CONTRASTS', len(results), flush=True)


if __name__ == '__main__':
    main()
