"""Plot only the normalized structural service-capacity model."""
import argparse
from poc_ouro_parallelism_deep_discovery.contract import ROOT, read
from poc_ouro_parallelism_deep_discovery.scripts.analysis_and_plots import svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    directory = ROOT/'analysis'/args.source
    records = read(directory/'screen.json')['records']
    for size in ['1.4B', '2.6B']:
        output = directory/f'{size}_structural_capacity.svg'
        assert not output.exists()
        series = [dict(name=f"{row['dataset']} P{row['P']}",
                       points=[(value['C'], value['pinned_over_pooled_capacity']) for value in row['finite_C']])
                  for row in records if row['model'] == size and row['P'] in [2, 4]]
        svg(output, series, f'{size}: unit-cost depth-affinity model, NOT measured TPS',
            'System-wide C', 'Pinned / ideal pooled service capacity')


if __name__ == '__main__':
    main()
