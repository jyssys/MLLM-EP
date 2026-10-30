"""Pinned runtime reuse, separate evidence, and a hard experiment deadline."""
import json
import os
import time
from datetime import datetime
from pathlib import Path

from ..looped_pd_abc.common import MODEL_PINS, MODELS, PYTHON, REPO, RLT, RLT_REVISION, UUIDS, digest
from ..looped_pd_abc.common import write, environment

ROOT = REPO / 'poc_flashvep/reports/looped_pd_b_slo_serving'
START = datetime.fromisoformat('2026-10-29T21:29:30+09:00').timestamp()
HARD_STOP = START + 15 * 3600
GPU_STOP = HARD_STOP - 45 * 60
BASE = 'b82eabd20a4ec168a5ad9a119c8401f97a565e78'
MODELS_USED = ('ouro', 'huginn')


def setup():
    for name in ('raw', 'logs', 'configs', 'sources', 'figures', 'state', 'archives', 'data'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    clock = ROOT / 'configs/budget.json'
    if not clock.exists():
        write(clock, dict(start_epoch=START, hard_stop_epoch=HARD_STOP, GPU_stop_epoch=GPU_STOP,
                         start_iso='2026-10-29T21:29:30+09:00', hours=15,
                         reporting_reserve_minutes=45, prior_commit=BASE,
                         scope='All setup, qualification, calibration, serving, diagnosis, and reporting.'))


def remaining():
    return GPU_STOP - time.time()


def env(gpus):
    value = environment(gpus)
    value['ABC_ASYNC_EVIDENCE'] = '1'
    return value


def append(path, value):
    from ..looped_pd_abc.telemetry import append as nonblocking
    nonblocking(path, value)


def configure_reuse():
    """Change only process-local output paths. Preserve previous source and evidence."""
    from ..looped_pd_abc import common, worker, nvml_watch
    common.ROOT = ROOT
    worker.ROOT = ROOT
    worker.setup = setup
    return worker
