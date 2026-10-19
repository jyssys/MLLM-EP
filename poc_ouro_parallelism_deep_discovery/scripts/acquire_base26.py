"""Fetch one missing public checkpoint at its pinned revision into the shared cache."""
import datetime
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'ByteDance/Ouro-2.6B'
REVISION = '1ed04250da1a9936042725d302e81c8fa2ab5abd'

def main():
    artifact = ROOT/'artifacts/base26_acquisition.json'
    if artifact.exists():
        raise SystemExit('An acquisition record already exists. Inspect it before resuming.')
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    files = [{ 'name': row.rfilename, 'bytes': row.size } for row in info.siblings]
    metadata = dict(model=MODEL, revision=REVISION, start_utc=start, files=files,
        purpose='Both-size Base checkpoint audit and bounded component/correctness probes.',
        scope='Single controlled missing-checkpoint acquisition. Existing caches remain unchanged.')
    artifact.write_text(json.dumps(metadata, indent=2)+'\n')
    directory = snapshot_download(MODEL, revision=REVISION,
        allow_patterns=['config.json', 'configuration_ouro.py', 'modeling_ouro.py', 'model.safetensors',
                        'tokenizer*', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'README.md'],
        max_workers=1)
    metadata.update(local_path=directory, success=True,
        end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    artifact.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata))

if __name__ == '__main__': main()
