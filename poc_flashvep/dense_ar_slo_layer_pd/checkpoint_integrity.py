"""Check the cached original shards without changing model files."""
from .common import *


def metadata():
    rows=[]
    for path in sorted(MODEL.glob('*.safetensors')):
        stat=path.stat();blob=path.resolve().name
        rows.append(dict(file=path.name,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
            HF_blob_name=blob,expected_LFS_SHA256=blob if len(blob)==64 else None))
    assert rows,'No checkpoint shard found'
    return rows


def record():
    path=ROOT/'state/model_checkpoint_file_inventory.json';rows=metadata()
    if path.exists():assert json.loads(path.read_text())['shards']==rows
    else:write(path,dict(time=time.time(),model_revision=MODEL_REVISION,shards=rows,
        scope='Read-only cache metadata. The LFS blob name is an expected checksum, not a freshly computed checksum. Full shard hashing runs after primary GPU measurement closes.'))


def verify():
    assert (ROOT/'state/CAMPAIGN_CLOSED.json').exists()
    old=json.loads((ROOT/'state/model_checkpoint_file_inventory.json').read_text())
    assert old['shards']==metadata(),'Checkpoint file metadata changed during confirmation'
    rows=[]
    for item in old['shards']:
        actual=digest(MODEL/item['file']);expected=item['expected_LFS_SHA256']
        assert expected is not None and actual==expected,'Original checkpoint shard checksum mismatch'
        rows.append(dict(file=item['file'],SHA256=actual,expected_HF_blob_SHA256=expected,passed=True))
    write(ROOT/'state/model_checkpoint_integrity_verified.json',dict(time=time.time(),passed=True,
        model_revision=MODEL_REVISION,shards=rows,evidence='MEASURED',
        scope='CPU byte checksums of original cached safetensors match their content-addressed LFS blob names. This is not a GPU performance measurement.'))
    print('ORIGINAL_MODEL_SHARDS_VERIFIED',len(rows),flush=True)


if __name__=='__main__':record()
