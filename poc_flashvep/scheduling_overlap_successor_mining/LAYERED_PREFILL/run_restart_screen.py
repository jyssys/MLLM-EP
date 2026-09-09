"""Paired official graph-enabled chunk/group screen; no new scheduler."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {'chunk512': ('chunked-prefill', 512, 16),
           'chunk2048': ('chunked-prefill', 2048, 16),
           'layered4': ('layered-prefill', 8192, 4),
           'layered12': ('layered-prefill', 8192, 12),
           'layered16': ('layered-prefill', 8192, 16),
           'layered24': ('layered-prefill', 8192, 24)}


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4,5,6,7'
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--restarts', type=int, default=3)
    p.add_argument('--workload-warmup-repeats', type=int, default=0)
    p.add_argument('--mechanism', action='store_true', help='Eager diagnostic; not clean graph-enabled performance')
    p.add_argument('--variants', nargs='+', choices=CONFIGS,
                   default=['chunk512', 'layered4', 'layered16'])
    args = p.parse_args()
    out = args.results/'layered_runs'/args.name
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(20260909)
    plan = []
    for block in range(args.restarts):
        variants = list(args.variants)
        rng.shuffle(variants)
        traces = ['heterogeneous_steady', 'heterogeneous_bursty']
        rng.shuffle(traces)
        for variant in variants:
            plan.append(dict(block=block, variant=variant, traces=list(traces)))
    manifest = dict(plan=plan, records=[], status='RUNNING',
                    workload_warmup_repeats=args.workload_warmup_repeats,
                    mechanism_instrumented=args.mechanism,
                    contract='independent paired restarts; official CUDA graphs; unchanged requests/arrivals')
    def save():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    save()
    for item in plan:
        mode, tokens, stages = CONFIGS[item['variant']]
        name = f"b{item['block']}_{item['variant']}"
        target = out/name
        smoke = args.results/'request_traces/qwen3_smoke.jsonl'
        command = [sys.executable, str(ROOT/'LAYERED_PREFILL/run_configuration.py'),
                   '--mode', mode, '--tokens', str(tokens), '--stages', str(stages),
                   '--out', str(target), '--warmup', str(smoke),
                   '--correctness-trace', str(smoke),
                   '--workload-warmup-repeats', str(args.workload_warmup_repeats)]
        command += ['--mechanism'] if args.mechanism else ['--cuda-graph']
        for trace in item['traces']:
            command += ['--trace', str(args.results/'request_traces/text_v1'/(trace+'.jsonl'))]
        print('START', name, flush=True)
        with (out/(name+'.log')).open('w') as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=2700)
        record = dict(item, name=name, command=command, returncode=run.returncode)
        manifest['records'].append(record)
        save()
        print('DONE', name, run.returncode, flush=True)
        if run.returncode:
            manifest['status'] = 'STOPPED_ON_BASELINE_FAILURE'
            save()
            raise SystemExit(run.returncode)
        meta = json.loads((target/'run.json').read_text())
        for measured in meta['request_runs']:
            if not measured['label'].startswith('measure'):
                continue
            subprocess.run([sys.executable, str(ROOT/'COMMON/record_gpu_interval.py'),
                            '--log', str(ROOT/'GPU_TIME_LOG.csv'),
                            '--run-id', args.name+':'+measured['run_id'],
                            '--candidate', 'LAYERED_PREFILL', '--gpus', '4,5',
                            '--question', 'Official chunk/group request-level envelope after common warmup',
                            '--start', str(measured['start_unix']),
                            '--end', str(measured['start_unix']+measured['elapsed_s']),
                            '--kind', 'mechanism_diagnostic' if args.mechanism else 'measurement',
                            '--valid', 'INSTRUMENTED_EAGER_NOT_PERFORMANCE' if args.mechanism else 'CROSS_CONFIG_CORRECTNESS_PENDING',
                            '--artifact', str(target/(measured['label']+'.jsonl'))], check=True)
    manifest['status'] = 'COLLECTED_CROSS_CONFIG_CORRECTNESS_PENDING'
    save()


if __name__ == '__main__':
    main()
