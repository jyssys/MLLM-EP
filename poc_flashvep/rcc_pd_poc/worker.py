"""Fixed-cut controls on the qualified runtime. Native Mixed remains a baseline."""
import json
import os
import sys
import time
from pathlib import Path
from .common import ROOT, CUTS, MODEL_PINS, setup, write, append


def install():
    from ..looped_pd_abc import common
    common.ROOT = ROOT
    common.setup = setup
    common.append = append
    from .runtime_contract import install as qualified
    qualified()
    from .probe import install as probe
    probe()
    from .state_probe import install as state_probe
    state_probe()


if os.environ.get('RCC_RUN'):
    install()


def serve(engine, inputs, args, run, warmup=False):
    from vllm_rlt import SamplingParams
    contract = json.loads(Path(os.environ['RCC_CONFIG']).read_text())
    offsets = contract['arrival_offsets'][:len(inputs)] if not warmup else [0.] * len(inputs)
    start = time.monotonic()
    active, outputs, first, last, events, admissions = {}, {}, {}, {}, {}, {}
    next_index = 0
    while next_index < len(inputs) or active:
        now = time.monotonic()
        while next_index < len(inputs) and now - start >= offsets[next_index]:
            index = next_index
            rid = f'r{run}_q{index}'
            seed = contract['seed'] + index
            params = SamplingParams(max_tokens=args.cap, temperature=0., seed=seed,
                                    min_loops=MODEL_PINS[args.model][2], max_loops=MODEL_PINS[args.model][2],
                                    exit_threshold=1., ignore_eos=contract.get('ignore_eos', False))
            engine.add_request(rid, inputs[index], params)
            active[rid] = index
            admissions[rid] = time.monotonic()
            events[rid] = []
            next_index += 1
        for output in engine.step():
            now = time.monotonic()
            rid = output.request_id
            count = len(output.token_ids)
            old_time, old_count = last.get(rid, (now, 0))
            if count > old_count:
                first.setdefault(rid, now)
                events[rid].append(dict(time=now, previous_time=old_time,
                                        new_tokens=count-old_count, first=old_count == 0))
                last[rid] = now, count
            if not output.finished:
                continue
            index = active.pop(rid)
            if any(depth != MODEL_PINS[args.model][2] for depth in output.exit_depths):
                raise RuntimeError('The prescribed recurrent depth changed')
            arrival = start + offsets[index]
            row = dict(model=args.model, arm=contract['arm'], run=run, sample_id=index,
                       input_ids=inputs[index], output_ids=list(output.token_ids),
                       input_tokens=len(inputs[index]), output_tokens=count,
                       exit_depths=list(output.exit_depths), finish_reason=str(output.finish_reason),
                       seed=contract['seed']+index, arrival=arrival, admission=admissions[rid], finish=now,
                       ttft_ms=1000*(first.get(rid, now)-arrival),
                       tpot_ms=1000*(now-first.get(rid, now))/max(count-1, 1),
                       e2e_ms=1000*(now-arrival), token_events=events[rid],
                       warmup=warmup, gpus=args.gpus, depth=MODEL_PINS[args.model][2],
                       evidence='MEASURED_DIAGNOSTIC' if any(contract.get(k) for k in ('probe','verify_kv','profile','state_probe')) else 'MEASURED',
                       generation='Controlled ignore-EOS' if contract.get('ignore_eos') else 'Natural EOS with a bounded output cap',
                       prompt_source='Frozen ShareGPT token-prefix mechanism probe. Not a representative W1 serving trace.')
            append(ROOT/'raw'/args.name/'requests.jsonl', row)
            outputs[index] = row
            print('DONE', args.name, run, index, count, str(output.finish_reason), flush=True)
        if time.monotonic()-start > args.timeout:
            raise TimeoutError('The RCC mechanism cohort exceeded its watchdog')
        if not active and next_index < len(inputs):
            time.sleep(min(.001, max(0., start+offsets[next_index]-time.monotonic())))
    if not warmup:
        write(ROOT/'raw'/args.name/'window.json', dict(start=start, end=time.monotonic(),
                                                      seconds=time.monotonic()-start,
                                                      scope='Small mechanism cohort, not sustainable serving capacity'))
    return outputs


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    if cfg['arm'] not in ('PD_G1', 'A_sep', 'A_native_mixed'):
        raise ValueError('No unqualified RCC or adaptive arm is available')
    install()
    from ..looped_pd_abc import worker as base
    base.ROOT = ROOT
    base.setup = setup
    base.append = append
    base.serve = serve
    data = json.loads((ROOT/'data'/cfg['dataset']).read_text())
    base.prompts = lambda model, count, length: [item['input_ids'][model] for item in data[:count]]
    old_configuration = base.configuration

    def configuration(model, width, chunk, graphs=False, asynchronous=False):
        value = old_configuration(model, width, chunk, graphs, asynchronous)
        from dataclasses import replace
        value['execution_config'] = replace(value['execution_config'],
                                            cuda_graph_max_batch_size=cfg.get('graph_rows', width))
        return value

    base.configuration = configuration
    write(ROOT/'raw'/cfg['name']/'rcc_contract.json', cfg)
    flags = ['worker', '--model', cfg['model'], '--mode', 'pd', '--gpus', ','.join(map(str, cfg['gpus'])),
             '--name', cfg['name'], '--requests', str(cfg['count']), '--length', '0',
             '--cap', str(cfg['cap']), '--chunk', str(cfg['chunk']), '--concurrency', str(cfg['capacity']),
             '--prefill', '1', '--timeout', str(cfg.get('timeout', 600)), '--wavefront']
    if cfg.get('graphs', True):
        flags.append('--graphs')
    if cfg.get('warmup', True):
        flags.append('--warmup')
    if cfg.get('profile'):
        flags.append('--profile')
    if cfg['model'] == 'ouro' and cfg.get('asynchronous', True):
        flags.append('--asynchronous')
    if cfg['arm'] != 'PD_G1':
        flags += ['--enable-recurrence-split', '--split-cut', str(CUTS[cfg['model']])]
    elif cfg['model'] == 'huginn' or cfg.get('verify_kv'):
        flags.append('--enable-boundary-adapter')
    if cfg['arm'] == 'A_native_mixed':
        flags += ['--scheduler-mode', 'mixed']
    sys.argv = flags
    base.main()


if __name__ == '__main__':
    main()
