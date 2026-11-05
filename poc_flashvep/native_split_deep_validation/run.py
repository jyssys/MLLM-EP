"""Reuse qualified execution and preserve the reused native adapter sources."""
import argparse
import zipfile
from .common import *


def execute(name):
    initialize()
    from ..native_loop_2d_oracle import run as inherited
    inherited.initialize = initialize
    from ..looped_pd_autonomous_headroom import run as native
    original = native.environment
    def environment(gpus):
        env = original(gpus)
        folder = ROOT / 'raw' / name
        paths = sorted((REPO / 'poc_flashvep/native_loop_2d_oracle').rglob('*.py'))
        with zipfile.ZipFile(folder / 'native_adapter_source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, str(path.relative_to(REPO)))
        write(folder / 'native_adapter_source_hashes.json', {str(path): digest(path) for path in paths})
        return env
    native.environment = environment
    try:
        return inherited.execute(name)
    finally:
        native.environment = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    raise SystemExit(0 if execute(args.name) else 1)
