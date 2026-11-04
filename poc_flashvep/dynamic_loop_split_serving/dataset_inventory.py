"""Count preserved token IDs and sources without claiming natural output lengths."""
from collections import Counter
from .common import *


def main():
    initialize()
    used = {}
    for path in (ROOT / 'configs').glob('*.json'):
        cfg = json.loads(path.read_text())
        if isinstance(cfg, dict) and cfg.get('dataset') and cfg.get('model') in ('ouro', 'huginn'):
            used.setdefault((cfg['model'], cfg['dataset']), []).append(cfg)
    rows = []
    for (model, dataset), configurations in sorted(used.items()):
        path = ROOT / 'data' / dataset
        pool = json.loads(path.read_text())
        lengths = [len(row['input_ids'][model]) for row in pool]
        prefix = pool[:128]
        assert all(length == row['length'] for length, row in zip(lengths, pool))
        if model == 'huginn':
            assert all(max(lengths) + cfg['cap'] <= 4096 for cfg in configurations)
        rows.append(dict(model=model, dataset=dataset, SHA256=digest(path), records=len(pool),
            unique_prompt_IDs=len({row['prompt_id'] for row in pool}),
            first_128_or_fewer_records=len(prefix),
            first_128_or_fewer_unique_prompt_IDs=len({row['prompt_id'] for row in prefix}),
            first_128_or_fewer_source_counts=dict(Counter(row['source_dataset'] for row in prefix)),
            token_ID_length_counts=dict(Counter(lengths)),
            source_dataset_counts=dict(Counter(row['source_dataset'] for row in pool)),
            source_type_counts=dict(Counter(row['source_type'] for row in pool)),
            explicit_truncation_records=sum(bool(row.get('truncation_explicit')) for row in pool),
            controlled_output_caps=sorted({cfg['cap'] for cfg in configurations}),
            scope='Preserved model-specific token IDs and original source hashes. Prefix truncation is explicit. '
                'Controlled ignore-EOS caps are not natural response lengths or Azure requests. '
                'The N512 trace, if executed, cycles the same 128-record pool.'))
    write_derived(ROOT / 'dataset_inventory.json', dict(datasets=rows,
        scope='Actual token-ID lengths, not a new tokenizer or source-text reproduction test. Native prefix caching remains separately audited.'))


if __name__ == '__main__':
    initialize()
    from ..looped_pd_autonomous_headroom.analysis_lane import quiet_analysis
    with quiet_analysis('preserved_dataset_inventory'):
        main()
