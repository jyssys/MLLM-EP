"""Frozen scope. Existing engines and earlier evidence remain unchanged."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
ASSETS=Path('/home/esjung/external/looped-exact-depth-chunk-assets')
SPEC=Path('/home/esjung/MLLM-EP-github/poc_flashvep/reports/looped_exact_depth_chunk_pipeline_poc_spec_v2_quality.md')
PYTHON='/home/esjung/.venvs/cp-composition-vllm026/bin/python'
CPU='/home/esjung/.venvs/llada2-mini-reference/bin/python'
GPUS=[4,5,6,7]
UUIDS=['6076e2f2-5b63-3761-5586-56ceb7df8139','a1a1cfcf-93a1-3544-9a5e-e58144b68730','e3f3998e-0f1a-e94a-b97c-4abb0e8c2c28','4cc26b88-19fc-1988-f9e0-17858aa7a99b']
REVISION='f1edd81e7ac41355db670500ceaf204e0f73af68'
MODEL=Path('/home/esjung/.cache/huggingface/hub/models--ByteDance--Ouro-2.6B-Thinking/snapshots')/REVISION
SEEDS=[41015,41016,41017]
DATASETS={'aime24':dict(repo='HuggingFaceH4/aime_2024',revision='2fe88a2f1091d5048c0f36abc874fb997b3dd99a',file='data/train-00000-of-00001.parquet'),
 'math100':dict(repo='HuggingFaceH4/MATH-500',revision='6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be',file='test.jsonl')}
TOLERANCE=dict(logits_max_abs=.125,hidden_relative_L2=.01,KV_relative_L2=.01,greedy_top1_equal=True,
    scope='HF BF16 component chunk-shape comparison, fixed before measurement. No relaxation after failure.')

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):digest.update(data)
    return digest.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')

def read(path):return json.loads(Path(path).read_text())
