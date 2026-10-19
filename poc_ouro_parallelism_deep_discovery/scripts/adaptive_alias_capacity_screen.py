"""Quantify an ideal terminal-alias payload model, not executable memory savings."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import statistics

from poc_ouro_parallelism_deep_discovery.contract import CDB, ROOT, read, sha, write
from poc_ouro_parallelism_deep_discovery.scripts.score_quality import interval


def payload_model(prompt_positions, depths, physical_layers):
    if type(prompt_positions) is not int or prompt_positions < 1 or physical_layers not in (24, 48):
        raise ValueError('Require a nonempty full-R4 prompt and a loaded Ouro layer count.')
    if not depths or any(type(value) is not int or value not in (2, 3, 4) for value in depths):
        raise ValueError('Require observed R2/R3/R4 decode positions.')
    per_depth = physical_layers*2*16*128*2
    full = 4*(prompt_positions+len(depths))*per_depth
    distinct = (4*prompt_positions+sum(depths))*per_depth
    return dict(prompt_positions=prompt_positions, recorded_decode_positions=len(depths),
                bytes_per_depth_position=per_depth, full_depth_payload_bytes=full,
                ideal_distinct_payload_bytes=distinct,
                ideal_payload_saving_percent=100*(full-distinct)/full,
                decode_only_payload_saving_percent=100*(4*len(depths)-sum(depths))/(4*len(depths)))


def sealed_page_model(prompt_positions, depths, page_size):
    if page_size < 1 or not depths or prompt_positions < 1:
        raise ValueError('Require positive page size, prompt and decode positions.')
    exits = [4]*prompt_positions+list(depths)
    if any(value not in (2, 3, 4) for value in exits):
        raise ValueError('Require R2/R3/R4 exits.')
    aliases = 0
    for start in range(0, len(exits)-page_size+1, page_size):
        page = exits[start:start+page_size]
        for target_depth in range(1, 5):
            source_depths = {min(target_depth, value) for value in page}
            if len(source_depths) == 1 and next(iter(source_depths)) < target_depth:
                aliases += 1
    total_depth_pages = 4*((len(exits)+page_size-1)//page_size)
    return dict(page_size=page_size, aliased_sealed_depth_pages=aliases,
                full_depth_pages=total_depth_pages, ideal_page_payload_saving_percent=100*aliases/total_depth_pages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = ROOT/'analysis/baseline_extension_complete_v1/summary.json'
    admitted = read(source)
    inputs = {str(source.relative_to(ROOT)): sha(source)}
    records = []
    for row in admitted['records']:
        if row['mode'] != 'adaptive':
            continue
        path = ROOT/row['source']
        if sha(path) != admitted['input_sha256'][row['source']]:
            raise ValueError('Admitted raw depth trace changed.')
        inputs[row['source']] = sha(path)
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            raw = json.load(stream)
        depths = [event['depth'] for event in raw['depth_events']]
        if len(depths) != row['recorded_decode_positions'] or sum(depths) != row['recorded_decode_core_pass_work']:
            raise ValueError('Depth counters disagree.')
        records.append(dict(size=row['size'], dataset=row['dataset'], id=row['id'],
            **payload_model(len(raw['prompt_ids']), depths, 24 if row['size'] == '1.4B' else 48),
            sealed_page_models=[sealed_page_model(len(raw['prompt_ids']), depths, page) for page in (16, 32, 256)]))
    summaries = []
    for size, domain in sorted({(r['size'], r['dataset']) for r in records}):
        rows = [r for r in records if (r['size'], r['dataset']) == (size, domain)]
        if len(rows) != 32 or len({r['id'] for r in rows}) != 32:
            raise ValueError('Require32 unique requests per cell.')
        savings = [r['ideal_payload_saving_percent'] for r in rows]
        summaries.append(dict(size=size, dataset=domain, N=32,
            mean_request_payload_saving_percent=statistics.mean(savings),
            request_bootstrap_CI_percent=interval(savings),
            total_payload_saving_percent=100*(1-sum(r['ideal_distinct_payload_bytes'] for r in rows)/
                                               sum(r['full_depth_payload_bytes'] for r in rows)),
            mean_decode_only_payload_saving_percent=statistics.mean(r['decode_only_payload_saving_percent'] for r in rows),
            sealed_page_sensitivity=[dict(page_size=page,
                mean_request_page_payload_saving_percent=statistics.mean(next(p for p in r['sealed_page_models'] if p['page_size'] == page)['ideal_page_payload_saving_percent'] for r in rows))
                for page in (16, 32, 256)]))
    args.output.mkdir(parents=True)
    with (args.output/'requests.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    sources = ['src/looped_cdb/kv_cache_policy.py', 'src/looped_cdb/continuous_depth_batching/cache.py',
               'src/looped_cdb/continuous_depth_batching/config.py',
               'src/looped_cdb/continuous_depth_batching/continuous_api.py']
    write(args.output/'summary.json', dict(records=records, summaries=summaries, input_sha256=inputs,
        source_sha256={relative: sha(CDB/relative) for relative in sources},
        evidence_class='ANALYTIC_PAYLOAD_MODEL_FROM_OBSERVED_DEPTHS',
        source_extents=['kv_cache_policy.py:1-260', 'cache.py:85-140', 'continuous_api.py:1198-1244', 'config.py:50-65', 'runtime.py:28-70'],
        scope='Committed historical KV payload at the last recorded decode position. FullR4 prompt plus observed decode passes. Ideal row aliases eliminate deeper terminal copies without collapsing distinct computed depth states.',
        excluded=['Page granularity', 'Alias metadata and lookup', 'Graph/static allocation', 'Transient live writes',
                  'Allocator slack', 'Weights', 'Scratch', 'Publication and lifetime costs'],
        qualification='No actual allocated/reserved HBM, HBM traffic, executable admission, timing, or new novelty claim. FixedR4 has zero saving in this model.',
        page_model='Alias only sealed whole pages whose occupied rows all select one shallower depth for the target slot. The final partial page never aliases. Page16 is the inspected config default, not a newly measured allocator. Page32/256 are modeled sensitivities. This restriction does not bound a finer-grained implementation.',
        candidate_effect='C03 scope supplement only. No finalist or runtime implementation.'))
    print(json.dumps(summaries))


if __name__ == '__main__':
    main()
