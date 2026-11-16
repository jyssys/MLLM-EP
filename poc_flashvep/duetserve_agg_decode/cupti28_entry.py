"""Add bounded API28 activity to diagnostic workers without changing model code."""
import json
import subprocess
import sys

from .common import ROOT


def worker_command(command):
    result = list(command)
    module = 'poc_flashvep.dense_ar_slo_layer_pd.native'
    if '--role' in result and '-m' in result and result[result.index('-m') + 1] == module:
        result[result.index('-m') + 1] = 'poc_flashvep.duetserve_agg_decode.cupti28_entry'
    return result


def main():
    from ..dense_ar_slo_layer_pd import native
    cfg = json.loads(open(sys.argv[1]).read())
    if '--role' in sys.argv:
        if cfg.get('api28_bounded_activity'):
            if cfg.get('api28_capture_mode') == 'PRECONTEXT_FULL':
                from ..dense_ar_layerwise_pd import activity as collector
            else:
                from ..dense_ar_elastic_layer_pd import background_activity as collector
            collector.ROOT = ROOT
            collector.install(cfg, sys.argv[sys.argv.index('--role') + 1])
        native.main()
        return
    original = subprocess.Popen
    def launch(command, *args, **kwargs):
        return original(worker_command(command), *args, **kwargs)
    subprocess.Popen = launch
    try:
        native.main()
    finally:
        subprocess.Popen = original


if __name__ == '__main__':
    main()
