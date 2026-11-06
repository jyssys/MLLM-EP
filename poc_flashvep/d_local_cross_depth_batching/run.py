"""Use the inherited exclusive GPU lease and bounded cleanup."""
import argparse
from .common import *


def execute(name):
    initialize()
    from ..native_split_deep_validation import run as inherited
    inherited.initialize = initialize
    import subprocess
    original = subprocess.Popen
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    def launch(command, *args, **kwargs):
        target = 'poc_flashvep.looped_pd_autonomous_headroom.operator_micro'
        if cfg.get('worker') == 'operator_micro' and isinstance(command, list) and target in command:
            command = list(command)
            command[command.index(target)] = 'poc_flashvep.d_local_cross_depth_batching.' + (
                'thin_selftest' if cfg.get('thin_selftest') else
                'slice_micro' if cfg.get('slice_micro') else
                'opportunity_micro' if cfg.get('micro_opportunity') else 'micro')
        return original(command, *args, **kwargs)
    subprocess.Popen = launch
    try:
        return inherited.execute(name)
    finally:
        subprocess.Popen = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    raise SystemExit(0 if execute(args.name) else 1)
