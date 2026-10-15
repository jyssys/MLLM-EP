"""Record content hashes. Check the immutable HF weight-blob SHA256 name."""
from poc_looped_exact_depth_chunk_pipeline.contract import *


def main():
    records=[]
    for path in sorted(MODEL.iterdir()):
        if not path.is_file():continue
        digest=sha(path);resolved=path.resolve()
        if path.suffix=='.safetensors':assert digest==resolved.name,'Pinned HF weight content does not match its blob SHA256'
        records.append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest,resolved_blob=str(resolved)))
    target=ROOT/'results/checkpoint_files.json'
    if target.exists():assert read(target)['files']==records,'Checkpoint files changed after the content audit'
    else:write(target,dict(revision=REVISION,files=records,scope='Content audit. No model or tokenizer file changes.'))
    print('PINNED_CHECKPOINT_FILES',len(records))


if __name__=='__main__':main()
