"""Reuse the guarded dispatcher and freeze both old and new adapters."""
import argparse
import fcntl
import os
import zipfile
from . import common
from ..duetserve_agg_decode import run as inherited


def execute(name, restore=False):
    common.initialize()
    from .runtime_freeze import check
    check()
    for key in ('BASE','MODEL_REVISION','PYTHON','QUIET_LOCK','REPO','ROOT',
                'digest','git','initialize','preserve','resources','write'):
        setattr(inherited, key, getattr(common, key))
    original = zipfile.ZipFile
    class Bundle(original):
        def write(self, filename, arcname=None, *args, **kwargs):
            if arcname=='sitecustomize.py':
                filename=common.SOURCE/'site_bootstrap.py'
            return super().write(filename, arcname, *args, **kwargs)
        def __exit__(self, *args):
            if self.mode=='w':
                for package in ('dense_ar_layer_token_pd','dense_ar_matched_axis_pd'):
                    for path in (common.REPO/'poc_flashvep'/package).glob('*.py'):
                        self.write(path,str(path.relative_to(common.REPO)))
            return super().__exit__(*args)
    zipfile.ZipFile=Bundle
    os.environ['DENSE_MATCHED_AXIS_BOOTSTRAP']='1'
    try:
        result=inherited.execute(name, restore=restore)
    finally:
        zipfile.ZipFile=original
    if result:
        import json
        folder=common.ROOT/'raw'/name
        cfg=json.loads((folder/'configuration.json').read_text())
        if cfg.get('auxiliary_only'):
            assert cfg.get('entry_module') in (
                'poc_flashvep.dense_ar_matched_axis_pd.isolated',
                'poc_flashvep.dense_ar_matched_axis_pd.isolated_serving_gemm')
            return result
        audits={p.name:json.loads(p.read_text()) for p in folder.glob('effective_runtime_*.json')}
        assert len(audits)==2
        passed=all(r['actual_scheduler_max_num_batched_tokens']==8192 and r['actual_scheduler_max_num_seqs']==1024 for r in audits.values())
        common.write(folder/'unified_worker_gate.json',dict(passed=passed,workers=audits,
            evidence='MEASURED',measurement_class='ACTUAL_STARTUP_CONFIGURATION'))
        assert passed, 'Actual workers differ from declared B/S'
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('names',nargs='+')
    args=parser.parse_args()
    try:
        for name in args.names:
            if not execute(name):
                break
    finally:
        with common.QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease,fcntl.LOCK_EX)
            if not common.resources().processes():
                common.resources().restore(list(range(8)))


if __name__=='__main__':
    main()
