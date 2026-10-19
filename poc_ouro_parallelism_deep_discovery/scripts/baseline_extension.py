"""Natural baseline robustness. No new engine, candidate, or gate is enabled."""
import argparse
import collections
import datetime as dt
import gzip
import json
import os
from pathlib import Path
import signal
import time
import traceback

from poc_ouro_parallelism_deep_discovery.contract import MODELS, ROOT, read, sha, validate_mask, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--mode', choices=('fixed', 'adaptive'), required=True)
    parser.add_argument('--protocol', default='artifacts/baseline_extension_protocol_v1.json')
    args = parser.parse_args()
    mask = validate_mask()
    assert len(mask) == 1
    protocol_path = ROOT / args.protocol
    protocol = read(protocol_path)
    assert protocol['status'] == 'FROZEN_BEFORE_NEW_OUTCOMES' and protocol['N'] == 64
    assert protocol['quality_cap'] == 8192 and protocol['graphs'] and protocol['batch'] == 1
    assert protocol['pins'][args.size]['revision'] == MODELS[args.size]['revision']
    old = read(ROOT / 'artifacts/probe_protocol.json')
    assert sha(ROOT / 'artifacts/probe_protocol.json') == protocol['existing_protocol_sha256']
    assert not {r['id'] for r in old['rows']} & {r['id'] for r in protocol['rows']}
    assert collections.Counter(r['dataset'] for r in protocol['rows']) == {'gsm8k': 32, 'math500': 32}
    output = Path(os.environ['LOOPED_KV_OUTPUT']) / 'baseline_extension'
    output.mkdir(parents=True, exist_ok=False)
    success, completed = False, []
    write(output / 'status.json', dict(success=False, phase='LOADING', N_expected=64))
    write(output / 'frozen_protocol.json', protocol)
    try:
        from poc_ouro_parallelism_deep_discovery.runtime import configure_imports, load_engine
        configure_imports()
        import torch
        from poc_decode_first_depth_aware_dp.runtime import stats_dict
        with torch.inference_mode():
            model, tokenizer, engine = load_engine(args.size, args.mode, graphs=True,
                                                   batch=1, blocks=640, max_model_len=10240)
            assert tokenizer.eos_token_id == protocol['EOS'] == 0
            engine.generate_batch([tokenizer.encode('Q: What is 1 + 1?\nA:')],
                max_new_tokens=4, eos_token_id=tokenizer.eos_token_id, warmup=True)
            depths, original = [], engine._record_exit
            def record(item, depth):
                depths.append(dict(position=item.token_position, depth=depth + 1))
                return original(item, depth)
            engine._record_exit = record
            def watchdog(signum, frame):
                raise TimeoutError('A baseline-extension request exceeded1200seconds')
            signal.signal(signal.SIGALRM, watchdog)
            for row in protocol['rows']:
                ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
                assert len(ids) + protocol['quality_cap'] <= 10240
                stops = [tokenizer.encode(value, add_special_tokens=False) for value in row.get('stop_strings', [])]
                depths.clear()
                torch.cuda.synchronize()
                start = time.perf_counter()
                signal.alarm(1200)
                try:
                    result = engine.generate_batch([ids], max_new_tokens=protocol['quality_cap'],
                        eos_token_id=tokenizer.eos_token_id, stop_sequences=stops, warmup=False)[0]
                finally:
                    signal.alarm(0)
                torch.cuda.synchronize()
                wall = (time.perf_counter() - start) * 1000
                tokens = list(result.generated_tokens)
                depth_count_matches = len(depths) == len(tokens) - 1
                depth_range_matches = all(value['depth'] in ([4] if args.mode == 'fixed' else [2, 3, 4]) for value in depths)
                reason = 'eos' if tokens and tokens[-1] == 0 else 'length' if len(tokens) == 8192 else 'stop'
                value = dict(id=row['id'], dataset=row['dataset'], size=args.size, mode=args.mode,
                    runtime='staged_reference_baseline_extension', model=MODELS[args.size]['id'],
                    revision=MODELS[args.size]['revision'], physical_gpu=mask[0], prompt_ids=ids,
                    output_ids=tokens, text=tokenizer.decode(tokens, skip_special_tokens=False),
                    generated_tokens=len(tokens), finish_reason=reason, depth_events=list(depths),
                    stats=stats_dict(engine.last_stats), wall_ms=wall, cap=8192, temperature=0, EOS=0,
                    timestamp_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    source_protocol_sha256=sha(protocol_path), evidence_class='QUALITY_ONLY',
                    depth_count_matches=depth_count_matches, depth_range_matches=depth_range_matches,
                    timing_scope='Baseline request wall in independent one-GPU jobs. Not candidate or same-P architecture speedup.',
                    first_output_source='Full R4 prefill. Recorded decode depths exclude the first output token.')
                path = output / (row['id'].replace('/', '_') + '.json.gz')
                with gzip.open(path, 'xt', encoding='utf-8') as stream:
                    json.dump(value, stream, separators=(',', ':'), allow_nan=False)
                    stream.write('\n')
                assert depth_count_matches and depth_range_matches, 'Raw request retained before depth validation failure'
                completed.append(row['id'])
                write(output / 'progress.json', dict(N_completed=len(completed), ids=completed,
                    size=args.size, mode=args.mode, held_out='UNTOUCHED'))
                print('BASELINE_EXTENSION', args.size, args.mode, row['id'], len(tokens), reason,
                      dict(collections.Counter(item['depth'] for item in depths)), flush=True)
            engine._record_exit = original
            success = True
    except BaseException as error:
        write(output / 'failure.json', dict(error_type=type(error).__name__, traceback=traceback.format_exc(),
                                            completed_ids=completed))
        raise
    finally:
        write(output / 'status.json', dict(success=success, N_completed=len(completed), N_expected=64,
            size=args.size, mode=args.mode, phase='COMPLETE' if success else 'FAILED',
            source_protocol_sha256=sha(protocol_path), candidate_quality='NOT_RUN', held_out='UNTOUCHED'))


if __name__ == '__main__':
    main()
