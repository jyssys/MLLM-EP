"""Pin and download the exact released artifacts. Never alter native files."""
import argparse
import json
import subprocess
from pathlib import Path
from .common import ASSETS, MODELS, ROOT, SPEC, XLLM, digest, record, write

def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser()
    p.add_argument('--download', action='store_true')
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    pins = {}
    existing = ROOT / 'model_pins.json'
    if existing.exists():
        pins = json.loads(existing.read_text())
    for name, repo in MODELS.items():
        if name not in pins:
            info = api.model_info(repo, files_metadata=True)
            pins[name] = dict(repo=repo, revision=info.sha,
                subfolder='Foil-1' if name == 'foil1' else None,
                files=[dict(path=f.rfilename, size=f.size) for f in info.siblings])
            write(existing, pins)
        if a.download:
            path = ASSETS / name
            record('DOWNLOAD_START', model=name, revision=pins[name]['revision'])
            snapshot_download(repo, revision=pins[name]['revision'], local_dir=path,
                allow_patterns=['Foil-1/*', 'README.md', 'LICENSE', 'NOTICE', 'modeling_loop_lm.py', 'loop_trace.py'] if name == 'foil1' else None,
                max_workers=2)
            record('DOWNLOAD_END', model=name, path=str(path))
    write(ROOT / 'source_lock.json', dict(spec=str(SPEC), spec_sha256=digest(SPEC),
        xllm_path=str(XLLM), xllm_revision=subprocess.check_output(['git', '-C', str(XLLM), 'rev-parse', 'HEAD'], text=True).strip(),
        models=pins))
    print(json.dumps({k: dict(repo=v['repo'], revision=v['revision']) for k,v in pins.items()}, indent=2))

if __name__ == '__main__':
    main()
