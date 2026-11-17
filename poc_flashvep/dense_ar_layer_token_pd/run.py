"""Reuse the audited quiet-lane dispatcher with a new immutable source bundle."""
import argparse
import fcntl
import os
import json
import time
import zipfile
from . import common
from ..duetserve_agg_decode import run as previous


def execute(name, restore=False):
    common.initialize()
    receipt = common.ROOT / 'raw' / name / 'dispatcher_result.json'
    if not receipt.exists():
        progress = json.loads((common.ROOT / 'state/progress.json').read_text())
        if time.time() > progress['deadline_epoch'] - 4200:
            common.write(common.ROOT / 'state/not_run_budget' / (name + '.json'), dict(
                status='NOT_RUN', reason='Research budget reserves the final70 minutes for evidence, reports and remote publication', time=time.time()))
            return False
    for key in ('BASE', 'MODEL_REVISION', 'PYTHON', 'QUIET_LOCK', 'REPO', 'ROOT',
                'digest', 'git', 'initialize', 'preserve', 'resources', 'write'):
        setattr(previous, key, getattr(common, key))
    original = zipfile.ZipFile
    class Bundle(original):
        def write(self, filename, arcname=None, *args, **kwargs):
            if arcname == 'sitecustomize.py':
                filename = common.SOURCE / 'site_bootstrap.py'
            return super().write(filename, arcname, *args, **kwargs)

        def __exit__(self, *args):
            if self.mode == 'w':
                for path in common.SOURCE.glob('*.py'):
                    self.write(path, str(path.relative_to(common.REPO)))
                common.write(common.ROOT / 'raw' / name / 'new_source_hashes.json',
                    {str(p.relative_to(common.REPO)): common.digest(p) for p in common.SOURCE.glob('*.py')})
                config=json.loads((common.ROOT/'configs'/(name+'.json')).read_text())
                models={}
                for field in ('cost_model','flow_cost_model'):
                    if config.get(field):
                        path=common.ROOT/'configs'/config[field]
                        models[field]=dict(name=path.name,SHA256=common.digest(path),content=json.loads(path.read_text()))
                data=common.ROOT/'data'/config['dataset']
                common.write(common.ROOT/'raw'/name/'evidence_input_snapshot.json',dict(
                    configuration_SHA256=common.digest(common.ROOT/'configs'/(name+'.json')),
                    dataset_file=data.name,dataset_SHA256=common.digest(data),models=models,
                    SLO_SHA256=common.digest(common.ROOT/'configs/frozen_SLO_regimes.json')))
            return super().__exit__(*args)
    zipfile.ZipFile = Bundle
    os.environ['DENSE_LAYER_TOKEN_BOOTSTRAP'] = '1'
    try:
        return previous.execute(name, restore=restore)
    finally:
        zipfile.ZipFile = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+')
    args = parser.parse_args()
    try:
        for name in args.names:
            if not execute(name):
                break
    finally:
        with common.QUIET_LOCK.open('a+') as lease:
            fcntl.flock(lease, fcntl.LOCK_EX)
            if not common.resources().processes():
                common.resources().restore(list(range(8)))


if __name__ == '__main__':
    main()
