"""Use the existing immutable dispatcher, exclusive lane, and owned cleanup."""
import argparse
import zipfile
from .common import *


def execute(name):
    initialize()
    from ..dynamic_loop_split_serving import run as inherited
    from ..looped_pd_autonomous_headroom import run as native
    original = native.environment
    def environment(gpus):
        env = original(gpus)
        folder = ROOT / 'raw' / name
        paths = sorted((REPO / 'poc_flashvep/dynamic_loop_split_serving').rglob('*.py'))
        with zipfile.ZipFile(folder / 'qualified_serving_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, str(path.relative_to(REPO)))
        write(folder / 'qualified_serving_source_hashes.json', {str(p): digest(p) for p in paths})
        return env
    native.environment = environment
    try:
        return inherited.execute(name)
    finally:
        native.environment = original
        initialize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    raise SystemExit(0 if execute(args.name) else 1)
