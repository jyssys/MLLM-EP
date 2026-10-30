"""Plot measured diagnostic evidence. Do not imply a serving speedup."""
import json

from .common import ROOT
from .jobs import parity


def main():
    files = [ROOT / 'configs' / (name + '.json') for name in
             ('b_repair_original_causal_KV', 'b_repair_canonical_causal_KV')]
    if not all(path.exists() for path in files):
        return
    values = [json.loads(path.read_text()) for path in files]
    checks = [parity(value['left'], value['right']) for value in values]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    labels = ['Original', 'Canonical candidate']
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].bar(labels, [value['total_different_elements'] for value in values], color=['#b44c44', '#386da7'])
    axes[0].set_ylabel('Different BF16 KV elements (count)')
    axes[0].set_title('Actual historical R3 KV prefix')
    for index, value in enumerate(values):
        axes[0].annotate(f"{value['total_different_elements']:,}", (index, value['total_different_elements']),
                         xytext=(0, 5), textcoords='offset points', ha='center')
    axes[1].bar(labels, [check['exact'] for check in checks], color=['#b44c44', '#386da7'])
    axes[1].set_ylim(0, max(check['N'] for check in checks) + 1)
    axes[1].set_ylabel('Exactly matching output sequences (count)')
    axes[1].set_title('Actual paired natural generation')
    for index, check in enumerate(checks):
        axes[1].annotate(f"{check['exact']}/{check['N']}", (index, check['exact']),
                         xytext=(0, 5), textcoords='offset points', ha='center')
    figure.suptitle('Ouro R4: measured diagnostic evidence, not E2E speedup')
    figure.text(.5, .01, 'Target request3, current position183 excluded. Identical prompt and token history through output index124.\n'
                         'One instrumented pair per variant. No between-start confidence interval.', ha='center', fontsize=8)
    figure.tight_layout(rect=(0, .10, 1, .92))
    for suffix in ('svg', 'png'):
        figure.savefig(ROOT / 'figures' / ('b_repair_causal_KV_and_output.' + suffix), dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    main()
