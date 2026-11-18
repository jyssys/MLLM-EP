"""Freeze the new inference adapters before clean performance."""
import json
import time
from .common import ROOT,SOURCE,digest,write

FILES=('configuration.py','serving.py','worker.py','site_bootstrap.py',
       'segmented_source.py','profile_labels.py','layer_witness.py')


def check():
    revision3=(ROOT/'state/design_freeze_v3_before_performance.json').exists()
    path=ROOT/'state'/('frozen_new_inference_sources_v3.json' if revision3 else 'frozen_new_inference_sources.json')
    files=FILES+('common_credits.py',) if revision3 else FILES
    actual={name:digest(SOURCE/name) for name in files}
    if path.exists():
        assert json.loads(path.read_text())['SHA256']==actual,'Frozen inference source changed'
    else:
        write(path,dict(time=time.time(),SHA256=actual,
            scope='New inference adapters. Predecessor and installed runtime hashes have separate guards. Analysis, figures and publication code can change without changing the inference source. Each worker ZIP still preserves every executed source version.'))
    return actual


if __name__=='__main__':
    print(check())
