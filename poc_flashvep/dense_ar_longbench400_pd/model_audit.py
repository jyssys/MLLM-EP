"""Preserve the exact local model and tokenizer files without modifying them."""
import json
from .common import ROOT,MODEL,MODEL_REVISION,write,digest


def main():
    rows=[]
    for path in sorted(MODEL.iterdir()):
        if path.is_file():rows.append(dict(file=path.name,resolved_path=str(path.resolve()),
            bytes=path.stat().st_size,SHA256=digest(path)))
    value=dict(model='Qwen/Qwen3-8B',revision=MODEL_REVISION,snapshot=str(MODEL),files=rows,
        scope='Read-only full-file hashes of local model weights,tokenizer and configuration. No numerical conversion.')
    path=ROOT/'references/model_tokenizer_file_hashes_v1.json'
    if path.exists():assert json.loads(path.read_text())==value
    else:write(path,value)
    print('Hashed',len(rows),'files and',sum(r['bytes'] for r in rows),'bytes.',flush=True)


if __name__=='__main__':main()
