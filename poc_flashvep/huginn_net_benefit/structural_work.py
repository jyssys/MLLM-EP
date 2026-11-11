"""Account actual submitted prompt geometry. FLOPs are structural, not counters."""
import argparse
import math
from collections import defaultdict
from .common import *


def causal_pairs(start, count):
    if start < 0 or count < 0:
        raise ValueError('Causal positions and row counts must be nonnegative')
    end = start+count
    return (end*(end+1)-start*(start+1))//2


def dense_work(rows, blocks, hidden, intermediate, adapter=False, tile=256):
    # The qualified fix pads only MLP down projections and the adapter.
    per_block = 4*hidden*hidden+3*hidden*intermediate
    sensitive = blocks*hidden*intermediate+(2*hidden*hidden if adapter else 0)
    weights = blocks*per_block+(2*hidden*hidden if adapter else 0)
    ideal = 2*rows*weights
    padded = math.ceil(rows/tile)*tile if rows else 0
    return ideal, ideal+2*(padded-rows)*sensitive


def summarize(name):
    from ..looped_pd_abc.common import MODELS
    cfg = json.loads((ROOT/'configs'/(name+'.json')).read_text())
    assert cfg['model'] == 'huginn'
    model_path = MODELS['huginn']/'config.json'
    model = json.loads(model_path.read_text())
    hidden, intermediate = model['n_embd'], model['intermediate_size']
    before, core, after = (model[key] for key in ('n_layers_in_prelude',
        'n_layers_in_recurrent_block', 'n_layers_in_coda'))
    folder = ROOT/'raw'/name
    begin = json.loads((folder/'measurement_begin.json').read_text())['time']
    expected = set(range(cfg['seed'], cfg['seed']+cfg['count']))
    roles = {}
    for path in folder.glob('atomic_spans_*.json'):
        value = json.loads(path.read_text())
        batches = defaultdict(list)
        for row in value['prompt_spans']:
            if row['seed'] in expected and row['time'] >= begin:
                batches[row['time']].append(row)
        total = dict(prelude_rows=0, recurrent_rows=0, coda_rows=0,
            ideal_dense_FLOPs=0, qualified_padded_dense_FLOPs=0,
            ideal_causal_attention_FLOPs=0, prompt_submissions=len(batches))
        for rows in batches.values():
            assert len({row['loop_count'] for row in rows}) == 1
            loops = rows[0]['loop_count']
            real = sum(row['count'] for row in rows)
            prelude = sum(row['count'] for row in rows if row['phase'] in ('prefill', 'prefilling_prelude'))
            coda = sum(max(0, min(row['start']+row['count'], row['prompt_tokens']-1)-row['start'])
                for row in rows if loops and row['depth']+loops == cfg['depth'])
            total['prelude_rows'] += prelude
            total['recurrent_rows'] += real*loops
            total['coda_rows'] += coda
            for count, blocks, adapter, repetitions in ((prelude, before, False, 1),
                    (real, core, True, loops), (coda, after, False, 1)):
                ideal, padded = dense_work(count, blocks, hidden, intermediate, adapter)
                total['ideal_dense_FLOPs'] += repetitions*ideal
                total['qualified_padded_dense_FLOPs'] += repetitions*padded
            for row in rows:
                pairs = causal_pairs(row['start'], row['count'])
                layers = core*loops+(before if row['phase'] in ('prefill', 'prefilling_prelude') else 0)
                total['ideal_causal_attention_FLOPs'] += 4*hidden*pairs*layers
                if loops and row['depth']+loops == cfg['depth']:
                    n = max(0, min(row['start']+row['count'], row['prompt_tokens']-1)-row['start'])
                    total['ideal_causal_attention_FLOPs'] += 4*hidden*causal_pairs(row['start'], n)*after
        roles[value['role']] = total
    actual = {key:sum(role[key] for role in roles.values()) for key in ('prelude_rows','recurrent_rows','coda_rows')}
    expected_work = dict(prelude_rows=cfg['count']*cfg['length'],
        recurrent_rows=cfg['count']*cfg['length']*cfg['depth'],
        coda_rows=cfg['count']*(cfg['length']-1))
    result = dict(name=name, roles=roles, actual_logical_prompt_work=actual,
        expected_logical_prompt_work=expected_work, work_matches=actual == expected_work,
        model_config_sha256=digest(model_path), dimensions=dict(hidden=hidden,
            intermediate=intermediate, prelude_layers=before, core_layers=core, coda_layers=after),
        scope='Submitted prompt spans are observed. Dense FLOPs and ideal causal Attention FLOPs are structural calculations. Padded dense work includes the active 256-row down_norm_adapter fix. These are not hardware FLOP counters, HBM traffic, kernel times, or speedup. Embedding, normalization, activation, softmax, RNG, and first-output LM-head work are excluded. All output-token Decode work is excluded.')
    write(ROOT/'work_accounting'/(name+'.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    initialize()
    print(json.dumps([summarize(name) for name in args.names]))


if __name__ == '__main__':
    main()
