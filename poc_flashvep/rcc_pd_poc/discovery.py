"""Small, resume-safe fixed-cut discovery. This is not a capacity campaign."""
import argparse
import gzip
import json
import subprocess
import time
import zipfile
from pathlib import Path
from .common import ROOT, CF_ROOT, REPO, PYTHON, environment, setup, write, append, digest


def inputs(length):
    path = ROOT/'data'/f'prefix_{length}.json'
    if path.exists():
        return path.name
    original = CF_ROOT/'data/W1.json.gz'
    rows = json.load(gzip.open(original, 'rt'))
    rows = [r for r in rows if all(len(r['input_ids'][m]) >= length for m in ('ouro', 'huginn'))][:8]
    if len(rows) != 8:
        raise RuntimeError('Insufficient real prompt prefixes for this mechanism probe')
    write(path, [dict(prompt_id=r['prompt_id'], source_sha256=digest(original),
                     transformation=f'First {length} tokens. Mechanism-only excerpt, not a complete chat prompt.',
                     input_ids={m:r['input_ids'][m][:length] for m in ('ouro', 'huginn')}) for r in rows])
    return path.name


def launch(cfg):
    folder = ROOT/'raw'/cfg['name']
    status = folder/'status.json'
    if status.exists() and json.loads(status.read_text()).get('status') == 'COMPLETE':
        return dict(name=cfg['name'], status='SKIPPED_COMPLETE')
    if folder.exists():
        raise RuntimeError('A partial run requires a new name: '+cfg['name'])
    config = ROOT/'configs'/(cfg['name']+'.json')
    write(config, cfg)
    folder.mkdir(parents=True)
    with zipfile.ZipFile(folder/'rcc_sources.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in Path(__file__).parent.glob('*.py'):
            archive.write(path, str(path.relative_to(REPO)))
    env = environment(cfg['gpus'])
    env.update(RCC_RUN=cfg['name'], RCC_MODEL=cfg['model'], RCC_CONFIG=str(config),
               RCC_PROBE=str(int(cfg.get('probe', False))), ABC_LANE=cfg['lane'],
               RCC_TARGET_TRACE=str(int(cfg.get('target_trace', False))),
               RCC_STATE_PROBE=str(int(cfg.get('state_probe', False))))
    if cfg.get('verify_kv'):
        env['ABC_VERIFY_KV'] = '1'
    command = [PYTHON, '-m', 'poc_flashvep.rcc_pd_poc.worker', str(config)]
    watcher = None
    if cfg['lane'] == 'quiet_pair':
        from ..looped_pd_b_numerical_diagnosis.fabric_watch import FabricWatch
        watcher = FabricWatch(folder, cfg['gpus'])
        watcher.start()
    try:
        with (ROOT/'logs'/(cfg['name']+'.log')).open('x') as log:
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            append(ROOT/'command_manifest.jsonl', dict(name=cfg['name'], command=command, gpus=cfg['gpus'],
                                                      environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'RCC_RUN', 'RCC_MODEL', 'RCC_CONFIG', 'ABC_HUGINN_REFERENCE')},
                                                      pid=process.pid, config_sha256=digest(config), time=time.time()))
            process.wait()
    finally:
        if watcher:
            watcher.close()
    result = dict(name=cfg['name'], exit_code=process.returncode,
                  status=json.loads(status.read_text()).get('status') if status.exists() else 'NO_STATUS')
    append(ROOT/'state/discovery_events.jsonl', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=('ouro', 'huginn'), required=True)
    p.add_argument('--gpus', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--length', type=int, default=128)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--verify-kv', action='store_true')
    p.add_argument('--eager', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--target-trace', action='store_true')
    p.add_argument('--state-probe', action='store_true')
    p.add_argument('--cap', type=int, default=32)
    p.add_argument('--repeat', type=int, default=1)
    a = p.parse_args()
    setup()
    gate = json.loads((ROOT/'configs/prerequisite_gate.json').read_text())
    if gate['models'][a.model]['status'] != 'CORRECTNESS_READY':
        raise RuntimeError('The model-specific prerequisite is not ready')
    gpus = list(map(int, a.gpus.split(',')))
    if len(gpus) != 2 or len(set(gpus)) != 2 or not set(gpus) <= set(range(8)):
        raise ValueError('A mechanism run needs two distinct authorized GPUs')
    dataset = inputs(a.length)
    results = []
    for repeat in range(a.repeat):
        # Counterbalance arm order across independent starts.
        arms = ['PD_G1', 'A_sep', 'A_native_mixed']
        if repeat % 2:
            arms.reverse()
        for arm in arms:
            cfg = dict(name=f'{a.tag}_{a.model}_{arm}_{repeat}', model=a.model, arm=arm, gpus=gpus,
                       dataset=dataset, count=8, cap=a.cap, chunk=a.length, capacity=8,
                       arrival_offsets=[i*(.04 if a.model == 'ouro' else .15) for i in range(8)],
                       seed=2026103017, graphs=not a.eager, graph_rows=8,
                       warmup=True, probe=a.probe, verify_kv=a.verify_kv, profile=a.profile, target_trace=a.target_trace,
                       state_probe=a.state_probe,
                       lane='diagnostic' if a.probe or a.verify_kv or a.profile or a.state_probe else 'quiet_pair', timeout=900)
            result = launch(cfg)
            results.append(result)
            write(ROOT/'state'/(a.tag+'_'+a.model+'.json'), dict(results=results, status='RUNNING'))
            if result.get('exit_code', 0) != 0:
                write(ROOT/'state'/(a.tag+'_'+a.model+'.json'), dict(results=results, status='FAILED_RUN'))
                return
    write(ROOT/'state'/(a.tag+'_'+a.model+'.json'), dict(results=results, status='COMPLETE_EXECUTIONS_ONLY'))


if __name__ == '__main__':
    main()
