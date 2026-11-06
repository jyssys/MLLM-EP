"""Freeze paired request traces before GPU tests."""
import json
import random
import shutil
from .common import *


def configuration(model, method='native', count=128, seed=2026110701, rate=None, markers=False, verify=False):
    path = PREVIOUS / 'configs' / f'd_{model}_native_heldout_s1_off128.json'
    cfg = json.loads(path.read_text())
    cfg.update(name=f'local_{model}_{method}_n{count}_s{seed}_'+('on' if markers else 'off')+('_state' if verify else ''),
        phase='D_LOCAL_CROSS_DEPTH_POC', dlocal_campaign=True, dlocal_method=method,
        freeze_dlocal_sources=True,
        count=count, arrival_seed=seed, observe_service_timing=markers,
        verify_transfer=verify, state_probe=verify,
        request_caps=[cfg['cap']]*count, timeout=900 if verify else 600,
        startup_timeout=300, deep_serial_primary_warmup=True,
        deep_complete_D_geometry_warmup=not verify)
    cfg['arrival_rate'] = rate if rate is not None else (3.4569045 if model=='ouro' else 1.06129395274)
    rng = random.Random(seed)
    offsets = [0.0]
    for _ in range(count-1):
        offsets.append(offsets[-1]+rng.expovariate(cfg['arrival_rate']))
    cfg['arrival_offsets'] = offsets
    cfg['dataset'] = f'heldout_{model}_L{cfg["length"]}.json'
    cfg['prompt_ids'] = cfg['prompt_ids'][:count]
    cfg['data_scope'] = dict(pool_records=128, measured_records=count, cyclic_reuse=False,
        selection='Preserved controlled prompt pool. Exploratory, not fresh held-out confirmation.')
    if method != 'native':
        cfg.update(D_token_budget=0, dlocal_chunk=32, dlocal_token_budget=256,
            dlocal_quantum=1, dlocal_all=verify,
            deep_complete_D_geometry_warmup=False)
        if method != 'full':
            cfg.update(atomic_prompt=False,P_atomic_prompt=True,
                dlocal_assertion_scope='P stays full-prompt. D-local requests intentionally use bounded prompt slices.')
    return cfg


def prepare():
    initialize()
    for model in ('ouro', 'huginn'):
        path = PREVIOUS / 'data' / f'heldout_{model}_L{4096 if model=="ouro" else 2048}.json'
        target = ROOT / 'data' / path.name
        if not target.exists():
            shutil.copy2(path, target)
        assert digest(path) == digest(target)
    frozen = PREVIOUS / 'configs/frozen_SLO.json'
    if not (ROOT / 'configs/frozen_SLO.json').exists():
        shutil.copy2(frozen, ROOT / 'configs/frozen_SLO.json')
    spec = REPO / 'poc_flashvep/reports/ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching_poc_spec.md'
    if not (ROOT / 'state/original_spec.md').exists():
        shutil.copy2(spec, ROOT / 'state/original_spec.md')
        write(ROOT / 'state/original_spec_hash.json', dict(path=str(spec), SHA256=digest(spec)))
    for model in ('ouro', 'huginn'):
        cfg = configuration(model)
        if not (ROOT / 'configs' / (cfg['name'] + '.json')).exists():
            save(cfg)


if __name__ == '__main__':
    prepare()
