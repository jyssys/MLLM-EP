"""Reuse the qualified quiet-lane dispatcher with a complete source snapshot."""
import json
import os
import time
import zipfile
from . import common


def execute(name,restore=False):
    common.initialize();common.bind_legacy()
    assert time.time()<common.DEADLINE-3600, 'Stop new GPU starts and retain publication time'
    from ..dense_ar_matched_axis_pd.runtime_freeze import check
    check()
    from ..duetserve_agg_decode import run as dispatcher
    for key in ('BASE','MODEL_REVISION','PYTHON','QUIET_LOCK','REPO','ROOT','digest','git',
                'initialize','preserve','resources','write'):
        setattr(dispatcher,key,getattr(common,key))
    original=zipfile.ZipFile
    class Bundle(original):
        def write(self,filename,arcname=None,*a,**kw):
            if arcname=='sitecustomize.py':
                filename=common.SOURCE/'site_bootstrap.py'
            return super().write(filename,arcname,*a,**kw)
        def __exit__(self,*a):
            if self.mode=='w':
                for package in ('dense_ar_layer_token_pd','dense_ar_matched_axis_pd','dense_ar_axis_sweetspot'):
                    for path in (common.REPO/'poc_flashvep'/package).glob('*.py'):
                        self.write(path,str(path.relative_to(common.REPO)))
            return super().__exit__(*a)
    zipfile.ZipFile=Bundle
    os.environ['DENSE_MATCHED_AXIS_BOOTSTRAP']='1'
    try:
        complete=dispatcher.execute(name,restore=restore)
    finally:
        zipfile.ZipFile=original
    if complete:
        folder=common.ROOT/'raw'/name
        cfg=json.loads((folder/'configuration.json').read_text())
        if cfg.get('auxiliary_only'):
            common.write(folder/'unified_worker_gate.json',dict(passed=None,
                status='NOT_APPLICABLE_ISOLATED_OPERATOR',
                scope='This auxiliary process does not launch serving workers. Do not count it as B/S-qualified online performance.'))
            return complete
        audits={p.name:json.loads(p.read_text()) for p in folder.glob('effective_runtime_*.json')}
        passed=(len(audits)==2 and all(v['actual_scheduler_max_num_batched_tokens']==8192 and
            v['actual_scheduler_max_num_seqs']==1024 for v in audits.values()))
        common.write(folder/'unified_worker_gate.json',dict(passed=passed,workers=audits,
            evidence='MEASURED_ONLINE',measurement_class='ACTUAL_STARTUP_CONFIGURATION'))
        assert passed
    return complete
