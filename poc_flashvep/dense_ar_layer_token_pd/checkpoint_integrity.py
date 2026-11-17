"""Check the pinned original weight bytes without copying the checkpoint."""
import fcntl
import time
from .common import ROOT,MODEL,MODEL_REVISION,QUIET_LOCK,resources,digest,write


def main():
    target=ROOT/'state/model_weights_SHA256_verification.json'
    assert not target.exists(),'Preserve the existing checkpoint audit'
    with QUIET_LOCK.open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX)
        assert not resources().processes();resources().restore(list(range(8)))
        started=time.time();rows=[]
        for path in sorted(MODEL.glob('model-*.safetensors')):
            expected=path.resolve().name
            assert len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
            actual=digest(path);assert actual==expected,path.name
            rows.append(dict(name=path.name,bytes=path.stat().st_size,SHA256=actual,
                expected_HF_cache_blob_SHA256=expected,passed=True))
        assert len(rows)==5
        metadata={name:digest(MODEL/name) for name in ('config.json','model.safetensors.index.json',
            'tokenizer.json','tokenizer_config.json','generation_config.json')}
        write(target,dict(time=time.time(),elapsed_s=time.time()-started,model_revision=MODEL_REVISION,
            model_path=str(MODEL),weights=rows,metadata_SHA256=metadata,passed=True,
            scope='Read every byte of the five local pinned BF16 safetensors shards. Each SHA256 matches its content-addressed Hugging Face cache blob. No weight file or installed runtime is changed. The full checkpoint is not republished.'))
        print('PINNED_BF16_WEIGHTS_SHA256_PASS',len(rows),flush=True)


if __name__=='__main__':
    main()
