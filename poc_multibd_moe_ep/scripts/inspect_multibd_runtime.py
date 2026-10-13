"""CPU-only pinning, exact unsupported-path reproduction, metadata freeze."""
import ast
import dataclasses
import hashlib
import importlib.metadata
import json
import random
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from poc_multibd_moe_ep.contract import *


def command(argv):
    return subprocess.check_output(argv, text=True).strip()


def main():
    assert command(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD']) == UPSTREAM_SHA
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    assert info.sha == MODEL_REVISION
    # Metadata only; no model weights downloaded before bounded bridge checks.
    model = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
        allow_patterns=['config.json', 'configuration_llada2_moe.py',
                        '*token*', 'chat_template.jinja', '*.index.json', 'README.md']))
    cfg = json.loads((model/'config.json').read_text())
    files = ['diffulex/config.py', 'diffulex/model/llada2.py',
             'diffulex/moe/layer/ep_impl.py',
             'diffulex/moe/dispatcher/deepep_dispatcher.py',
             'diffulex/engine/dllm_block.py',
             'diffulex/strategy_template/multi_block/engine/model_runner.py',
             'diffulex/strategy_template/multi_block/engine/scheduler.py',
             'diffulex_bench/configs/experiment/llada2_mini.yml']
    write(ROOT/'artifacts/pins.json', dict(spec_sha256=sha(SPEC), upstream_sha=UPSTREAM_SHA,
        upstream_branch='mbd-lms', model_id=MODEL_ID, model_revision=MODEL_REVISION,
        model_path=str(model), file_sha256={p:sha(UPSTREAM/p) for p in files},
        model_files=[dict(name=f.rfilename, bytes=f.size) for f in info.siblings],
        model_config=cfg, prior_ep_commit='dc539d3222193ab0902e0a342679edc3610f54bb'))
    sys.path.insert(0, str(UPSTREAM))
    from diffulex.config import Config
    from diffulex.moe.dispatcher.deepep_dispatcher import _DEEP_EP_IMPORT_ERROR
    error = None
    try:
        Config(model=str(model), model_name='llada2_mini', decoding_strategy='multi_bd',
            tensor_parallel_size=4, expert_parallel_size=4, data_parallel_size=1,
            device_ids=[0,1,2,3], moe_dispatcher_backend='deepep', deepep_mode='normal')
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    write(ROOT/'artifacts/upstream_ep_guard.json', dict(error=error,
        deep_ep_import_error=str(_DEEP_EP_IMPORT_ERROR) if _DEEP_EP_IMPORT_ERROR else None))
    # Freeze official benchmark metadata before any model generation/routing.
    rng = random.Random(3407)
    frozen = []
    for domain, filename in [('humaneval_plus', 'HumanEval_llada2.json'),
                             ('mbpp_plus', 'MBPP_plus_llada2.json')]:
        data = json.loads((UPSTREAM/'diffulex_bench/data'/filename).read_text())
        if isinstance(data, dict):
            rows = [dict(v, source_key=k) if isinstance(v,dict) else dict(value=v,source_key=k)
                    for k,v in data.items()]
        else:
            rows = data
        rng.shuffle(rows)
        for i, row in enumerate(rows[:16]):
            frozen.append(dict(domain=domain, split='calibration' if i<4 else 'held_out',
                source_file=filename, source_sha256=sha(UPSTREAM/'diffulex_bench/data'/filename),
                selected_index=i, data=row))
    write(ROOT/'artifacts/frozen_manifest.json',dict(seed=3407,
        selection='seeded metadata-only, before model execution', rows=frozen))
    write(ROOT/'artifacts/design_freeze.json',dict(R=[1,2,3,4], block_size=32,
        sampling_mode='naive', DMax=False, temperature=0,
        add_block_threshold=.9, semi_complete_threshold=.9, accept_threshold=.95,
        quality_max_new_tokens=4096, smoke_max_new_tokens=64, max_nfe=1024,
        topology='dense TP4 / strided source-SP4 / physical EP4 / DP1 / PP1',
        backend='DeepEP normal', precision='BF16', mask=MASK,
        bridge_attempt_budget=2, request_bootstrap_replicates=2000,
        correctness_rel_l2_tolerance=.02, correctness_max_scaled_abs_tolerance=.04,
        held_out_tolerance_retuning=False,
        stage_order=['CPU state machine','physical MoE bridge correctness',
                     'full-model R1/R>1 tiny','trace parity','R sweep','quality'],
        main_statistics='request-level paired; no layer-level independent samples'))
    gpu = command(['nvidia-smi','--id='+MASK,
        '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader'])
    write(ROOT/'artifacts/environment.json', dict(gpus=gpu,
        topology=command(['nvidia-smi','topo','-m']), mask=MASK,
        processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                           '--format=csv,noheader']),
        mapping=[dict(logical=i,physical=g,uuid=UUIDS[i]) for i,g in enumerate(GPUS)],
        python=sys.executable, versions={k:importlib.metadata.version(k) for k in
            ['torch','deep-ep','sglang','vllm','triton','transformers']},
        worktree=command(['git','rev-parse','HEAD']),
        original_worktree_staged_diff_sha256=hashlib.sha256(subprocess.check_output(
            ['git','-C','/home/esjung/MLLM-EP-github','diff','--cached','--binary'])).hexdigest(),
        original_worktree_unstaged_diff_sha256=hashlib.sha256(subprocess.check_output(
            ['git','-C','/home/esjung/MLLM-EP-github','diff','--binary'])).hexdigest()))
    print(json.dumps(dict(model_path=str(model), guard=error, rows=len(frozen))))


if __name__ == '__main__':
    main()
