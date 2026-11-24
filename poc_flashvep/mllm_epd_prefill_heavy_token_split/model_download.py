"""Pin the official model and processor revision before downloading weights."""
import json
import dataclasses
import time
from pathlib import Path
from .common import ROOT,MODEL_ID,digest,write,initialize,preserve


def main():
    initialize()
    from huggingface_hub import HfApi,snapshot_download
    path=ROOT/'configs/model_revision.json'
    if path.exists():revision=json.loads(path.read_text())['revision']
    else:
        info=HfApi().model_info(MODEL_ID,files_metadata=True);revision=info.sha
        write(path,dict(time=time.time(),model=MODEL_ID,processor=MODEL_ID,revision=revision,
            siblings=[dict(name=s.rfilename,size=s.size,LFS=dataclasses.asdict(s.lfs)
                if dataclasses.is_dataclass(s.lfs) else s.lfs) for s in info.siblings],
            scope='Official model and processor share this immutable revision. No quantization,weight or processor changes.'))
    print('PINNED_MODEL_REVISION',revision,flush=True)
    directory=Path(snapshot_download(MODEL_ID,revision=revision,max_workers=4,
        allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','README.md','LICENSE']))
    rows=[dict(path=str(p),bytes=p.stat().st_size,SHA256=digest(p)) for p in sorted(directory.iterdir()) if p.is_file()]
    write(ROOT/'state/model_download_complete.json',dict(time=time.time(),passed=True,revision=revision,
        directory=str(directory),files=rows,config=json.loads((directory/'config.json').read_text()),
        processor_config=json.loads((directory/'preprocessor_config.json').read_text())))
    preserve();print('OFFICIAL_MODEL_DOWNLOAD_COMPLETE',directory,flush=True)


if __name__=='__main__':main()
