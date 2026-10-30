"""Reuse ownership validation, with a separate resource ledger and burner log."""
from .common import ROOT, REPO, setup
from ..looped_pd_abc import resources as prior

base = prior.base
base.ROOT = ROOT
base.PRIOR = REPO / 'poc_flashvep/reports/looped_pd_abc'
base.APPROVED_LOGS.update({ROOT / 'burner.log', base.PRIOR / 'burner.log'})


def stop():
    setup()
    base.stop(list(range(8)))


def restore():
    setup()
    base.restore(list(range(8)))
