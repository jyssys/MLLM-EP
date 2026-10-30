"""Arrival-driven serving harness on unchanged frozen RLT model operators."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

from .common import ROOT, MODEL_PINS, configure_reuse, setup, write, append, digest

CONFIG = None
DATA = None


def arrival_trace(count, rate, seed, burst=False):
    import random
    random = random.Random(seed)
    if rate <= 0:
        return [0.0] * count, 0.0
    offsets = []
    moment = 0.0
    # Predeclared alternating states. Each 40s high-rate state lasts 10s.
    while len(offsets) < count:
        current = rate * (3.0 if burst and moment % 40 >= 30 else 1.0)
        boundary = (int(moment / 40) * 40 + (30 if moment % 40 < 30 else 40)) if burst else float('inf')
        interval = random.expovariate(current)
        if moment + interval >= boundary:
            moment = boundary
            continue
        moment += interval
        offsets.append(moment)
    return offsets, moment + 1.0 / rate


def serve(engine, inputs, args, run, warmup=False):
    from vllm_rlt import SamplingParams
    values = DATA[:len(inputs)]
    seed = CONFIG['seed']
    offsets, window = arrival_trace(len(inputs), CONFIG.get('rate', 0), seed, CONFIG.get('workload') == 'W3')
    if warmup:
        offsets, window = [0.] * len(inputs), 0.
    dest = ROOT / 'raw' / args.name
    write(dest / ('warmup_trace.json' if warmup else 'arrival_trace.json'),
          dict(seed=seed, offsets=offsets, window_seconds=window, rate=CONFIG.get('rate', 0),
               prompt_ids=[value['prompt_id'] for value in values],
               burst='3x in seconds 30..40 of each 40s cycle' if CONFIG.get('workload') == 'W3' else None))
    active = {}
    final = {}
    events = {}
    first = {}
    last = {}
    admitted = {}
    start = time.monotonic()
    next_index = 0
    last_queue_sample = start
    while next_index < len(inputs) or active:
        now = time.monotonic()
        while next_index < len(inputs) and now - start >= offsets[next_index]:
            # Qualification is explicitly sequential. Open-loop arrivals are never blocked by local concurrency.
            if CONFIG.get('rate', 0) <= 0 and len(active) >= args.concurrency:
                break
            index = next_index
            rid = f'r{run}_q{index}'
            params = SamplingParams(max_tokens=args.cap, min_loops=MODEL_PINS[args.model][2],
                                    max_loops=MODEL_PINS[args.model][2], exit_threshold=1.,
                                    temperature=0., seed=seed + index)
            engine.add_request(rid, inputs[index], params)
            admitted[rid] = time.monotonic()
            active[rid] = index
            events[rid] = []
            next_index += 1
            now = time.monotonic()
        for output in engine.step():
            now = time.monotonic()
            rid = output.request_id
            count = len(output.token_ids)
            previous = last.get(rid, (now, 0))
            if count > previous[1]:
                is_first = previous[1] == 0
                first.setdefault(rid, now)
                events[rid].append(dict(time=now, previous_time=previous[0], new_tokens=count - previous[1], first=is_first))
                last[rid] = now, count
            if not output.finished:
                continue
            index = active.pop(rid)
            actual_depths = list(output.exit_depths)
            if any(depth != MODEL_PINS[args.model][2] for depth in actual_depths):
                raise RuntimeError('Prescribed fixed depth changed')
            arrival = start + offsets[index] if CONFIG.get('rate', 0) > 0 and not warmup else admitted[rid]
            row = dict(model=args.model, arm=CONFIG['arm'], workload=CONFIG['workload'], sample_id=index,
                       prompt_id=values[index]['prompt_id'], input_ids=inputs[index], output_ids=list(output.token_ids),
                       input_tokens=len(inputs[index]), output_tokens=count, exit_depths=actual_depths,
                       finish_reason=str(output.finish_reason), truncated=str(output.finish_reason) == 'length',
                       arrival=arrival, admission=admitted[rid], finish=now, first_token=first.get(rid, now),
                       ttft_ms=1000 * (first.get(rid, now) - arrival),
                       tpot_ms=1000 * (now - first.get(rid, now)) / (count - 1) if count >= 2 else 0.,
                       e2e_ms=1000 * (now - arrival), token_events=events[rid],
                       delivery_delay_ms=1000 * (admitted[rid] - arrival),
                       seed=seed + index, warmup=warmup, run=run, attempt_id=args.attempt_id,
                       model_revision=MODEL_PINS[args.model][1], depth=MODEL_PINS[args.model][2],
                       dtype='bfloat16', kv_layout='last_exited', gpus=args.gpus,
                       prefill_devices=args.prefill, method=args.mode, timing_scope='Scheduled arrival to final output; harness, not HTTP',
                       prompt_source='Pinned ShareGPT conversation text with official chat template',
                       enable_recurrence_split=False, enable_frontier_handoff=CONFIG['arm'] == 'B', placement='upstream',
                       engine_revision='6b164d3844d35ab8e753df1864f0c87dc180401a',
                       evidence='MEASURED_DIAGNOSTIC_QUALIFICATION' if CONFIG.get('diagnostic') else 'MEASURED_OPEN_LOOP_BOUNDED_NATURAL_GENERATION')
            append(dest / 'requests.jsonl', row)
            final[index] = row
            print('DONE', args.name, index, count, output.finish_reason, round(row['e2e_ms'], 2), flush=True)
        now = time.monotonic()
        if now - last_queue_sample >= .2:
            append(dest / 'queue.jsonl', dict(time=now, elapsed=now - start, warmup=warmup,
                                             arrived=next_index, finished=len(final), outstanding=len(active)))
            last_queue_sample = now
        if now - start > args.timeout:
            raise TimeoutError(f'Window watchdog: arrived={next_index}, completed={len(final)}, planned={len(inputs)}')
        if not active and next_index < len(inputs):
            time.sleep(min(.001, max(0., start + offsets[next_index] - time.monotonic())))
    if not warmup:
        end = time.monotonic()
        write(dest / 'window.json', dict(start=start, end=end, evaluation_end=start + window,
                                         window_seconds=window, actual_span_seconds=end - start,
                                         drain_seconds=max(0., end - start - window), arrivals=len(inputs),
                                         offered_rate=CONFIG.get('rate', 0), seed=seed,
                                         actual_trace_rate=len(inputs) / window if window else None,
                                         completion_scope='Natural EOS with an explicit output safety cap'))
    return final


def main():
    global CONFIG, DATA
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration')
    arg = parser.parse_args()
    CONFIG = json.loads(Path(arg.configuration).read_text())
    if CONFIG['model'] not in ('ouro', 'huginn'):
        raise ValueError('Only the two pinned dense models are in scope')
    if CONFIG['arm'] == 'GenericLayer':
        raise NotImplementedError('GenericLayer has structural analysis only. No live result may use this label.')
    if CONFIG['arm'] not in ('DP', 'DP_Mixed', 'PD', 'PD_G1', 'B', 'Native', 'HF'):
        raise ValueError('An unknown or excluded method was requested')
    setup()
    DATA = json.loads((ROOT / 'data' / (CONFIG['dataset'] + '.json')).read_text())[:CONFIG['count']]
    module = configure_reuse()
    module.prompts = lambda model, count, length: [row['input_ids'][model] for row in DATA[:count]]
    module.serve = serve
    old_configuration = module.configuration
    def configuration(model, width, chunk, graphs=False, asynchronous=False):
        value = old_configuration(model, width, chunk, graphs, asynchronous)
        # Identical fixed core/cache configuration in every arm. Only approved schedule/transfer flags differ.
        return value
    module.configuration = configuration
    gpus = CONFIG['gpus']
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(map(str, gpus)):
        raise ValueError('Physical GPU mask mismatch')
    dest = ROOT / 'raw' / CONFIG['name']
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / 'serving_contract.json', CONFIG)
    with zipfile.ZipFile(dest / 'followup_sources.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in Path(__file__).parent.glob('*.py'):
            archive.write(source, source.name)
    if CONFIG.get('reference'):
        from ..looped_pd_abc import reference_probe
        reference_probe.ROOT = ROOT
        reference_probe.prompts = module.prompts
        sys.argv = ['reference_probe', '--model', CONFIG['model'], '--name', CONFIG['name'],
                    '--requests', str(CONFIG['count']), '--length', '0', '--cap', str(CONFIG['cap'])]
        reference_probe.main()
        return
    flags = ['worker', '--model', CONFIG['model'], '--mode', 'dp' if CONFIG['arm'] in ('DP', 'DP_Mixed') else 'native' if CONFIG['arm'] == 'Native' else 'pd',
             '--gpus', ','.join(map(str, gpus)), '--name', CONFIG['name'], '--requests', str(CONFIG['count']),
             '--length', '0', '--cap', str(CONFIG['cap']), '--chunk', '256',
             '--rate', str(CONFIG.get('rate', 0)),
             '--concurrency', str(CONFIG.get('capacity', 8)), '--prefill', str(len(gpus) // 2),
             '--timeout', str(CONFIG.get('timeout', 1800)), '--warmup', '--graphs']
    if CONFIG['model'] == 'ouro':
        flags += ['--asynchronous']
    if CONFIG['arm'] == 'DP_Mixed':
        flags += ['--scheduler-mode', 'mixed']
    if CONFIG['arm'] in ('PD_G1', 'B', 'GenericLayer'):
        flags += ['--wavefront']
    if CONFIG['arm'] == 'B':
        flags += ['--enable-frontier-handoff']
    if CONFIG['arm'] not in ('Native', 'DP', 'DP_Mixed') and CONFIG['model'] == 'huginn':
        flags += ['--enable-boundary-adapter']
    sys.argv = flags
    module.main()


if __name__ == '__main__':
    main()
