"""CPU-only filesystem and read-only GPU audit before acquiring resources."""
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
from pathlib import Path

from poc_dispatch_combine_asymmetry.contract import ROOT, SPEC, MLLM, LLM, geometry, write


def command(*args):
    out = subprocess.run(args, text=True, capture_output=True)
    return {'command': args, 'returncode': out.returncode, 'stdout': out.stdout, 'stderr': out.stderr}


def main():
    import torch
    from safetensors import safe_open
    models = {}
    for name, location in [('mllm', MLLM), ('llm', LLM)]:
        path = Path(location)
        cfg = json.loads((path / 'config.json').read_text())
        index = json.loads((path / 'model.safetensors.index.json').read_text())
        shapes = {}
        for key, filename in index['weight_map'].items():
            if 'layers.0.mlp.experts' in key and ('gate' in key or 'down' in key):
                with safe_open(str(path / filename), framework='pt') as handle:
                    shapes[key] = handle.get_slice(key).get_shape()
                if len(shapes) >= 3:
                    break
        models[name] = {'path': location, 'revision': path.name,
            'geometry': geometry(cfg), 'config': cfg, 'sample_weight_shapes': shapes,
            'shards_exist': all((path / f).is_file() for f in set(index['weight_map'].values())),
            'weight_bytes': index['metadata']['total_size'],
            'config_sha256': hashlib.sha256((path/'config.json').read_bytes()).hexdigest()}
    apps = command('nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader')
    owners = {}
    for line in apps['stdout'].splitlines():
        try:
            pid = int(line.split(',')[1])
            proc = Path(f'/proc/{pid}')
            owners[pid] = {'uid': proc.stat().st_uid, 'cmdline': (proc/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                           'cwd': str((proc/'cwd').resolve())}
        except (ValueError, FileNotFoundError, PermissionError):
            pass
    sources = [SPEC, Path('poc_mllm_ep_communication_motivation/artifacts/manifests/frozen_manifest.json'),
        Path('/home/esjung/vllm-ep/run_utilize.sh'),
        Path('/home/esjung/anaconda3/envs/flashvep-poc/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/prepare_finalize/deepep_ht.py'),
        Path('/home/esjung/.venvs/flashvep-deepep-v020/lib/python3.12/site-packages/deep_ep/buffer.py')]
    write(ROOT/'artifacts/gpu_audit/environment.json', {
        'resolved_spec_path': str(SPEC), 'requested_spec_path_exists': False,
        'spec_sha256': hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        'models': models, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'vllm': metadata.version('vllm'), 'deep_ep': metadata.version('deep-ep'),
        'gpu': command('nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,clocks.sm,temperature.gpu', '--format=csv,noheader'),
        'topology': command('nvidia-smi', 'topo', '-m'), 'processes': apps, 'process_owners': owners,
        'git': command('git', 'rev-parse', 'HEAD'), 'branch': command('git', 'branch', '--show-current'),
        'physical_to_logical': dict(zip([0,1,4,5],range(4))),
        'groups': {'tp': [[0,1],[2,3]], 'dp': [[0,2],[1,3]], 'ep': [0,1,2,3]},
        'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})


if __name__ == '__main__':
    main()
