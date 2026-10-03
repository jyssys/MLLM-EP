import numpy as np
from poc_dllm_request_persistent_replica.scripts.pack_evidence import pack_dir,verify


def test_lossless_compact_raw(tmp_path):
    source=tmp_path/'request';source.mkdir();out=tmp_path/'packed';out.mkdir()
    (source/'a.json').write_text('{"real": 3.5}\n')
    np.savez_compressed(source/'x.npz',ids=np.arange(32,dtype=np.uint16).reshape(4,8),
        same=np.arange(32,dtype=np.uint16).reshape(4,8),empty=np.empty((0,8),dtype=np.float32))
    record=pack_dir(source,out)
    assert record['arrays']==3 and record['unique_arrays']==2
    restored=tmp_path/'restored';verify(out/'request',restored)
    assert (restored/'a.json').read_bytes()==(source/'a.json').read_bytes()
    with np.load(source/'x.npz') as a,np.load(restored/'x.npz') as b:
        for key in a.files:
            assert a[key].dtype==b[key].dtype and np.array_equal(a[key],b[key])
