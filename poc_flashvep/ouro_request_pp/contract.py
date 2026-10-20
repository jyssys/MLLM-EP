from pathlib import Path
import hashlib
import json
import os

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / 'poc_flashvep/reports/ouro_request_pp_discovery'
MODEL = Path('/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-1.4B/snapshots/574fa66cb8bf5abdc979642d01cf2b79b16bfab1')
REVISION = MODEL.name
GPUS = [4, 5, 6, 7]
UUIDS = ['6076e2f2-5b63-3761-5586-56ceb7df8139', 'a1a1cfcf-93a1-3544-9a5e-e58144b68730',
         'e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28', '4cc26b88-19fc-1988-f9e0-17858aa7a99b']
SPEC = Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/ouro_request_pipeline_pp_discovery_spec.md')
TOLERANCE = dict(logit_max_abs=.5, logit_mean_abs=.05, KV_relative_L2=.02,
                 hidden_relative_L2=.02, greedy_ID_mismatches=0)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def validate_mask():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '4,5,6,7'


def route(placement, depth):
    assert depth in [2, 3, 4]
    if placement == 'layer_pp4':
        return [(rank, loop, rank * 6, (rank + 1) * 6) for loop in range(depth) for rank in range(4)]
    assert placement == 'loop_pp4'
    return [(loop, loop, 0, 24) for loop in range(depth)]
