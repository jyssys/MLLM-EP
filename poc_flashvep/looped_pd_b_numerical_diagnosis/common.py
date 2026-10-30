"""Separate evidence namespace and frozen input references."""
from pathlib import Path
from ..looped_pd_abc.common import REPO, RLT, MODELS, MODEL_PINS, PYTHON, write, digest, environment

ROOT = REPO / 'poc_flashvep/reports/looped_pd_b_numerical_diagnosis'
PRIOR = REPO / 'poc_flashvep/reports/looped_pd_b_slo_serving'
SOURCE = Path(__file__).parent


def setup():
    for name in ('raw', 'configs', 'logs', 'state', 'data'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def redirect():
    """Redirect reused harness outputs in this process and its spawned children."""
    from ..looped_pd_b_slo_serving import common
    common.ROOT = ROOT
    common.setup = setup
    from ..looped_pd_b_slo_serving import worker
    worker.ROOT = ROOT
    worker.setup = setup
    return worker
