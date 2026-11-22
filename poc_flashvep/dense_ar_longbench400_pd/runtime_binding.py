"""Bind reused observers to this process-local namespace without editing them."""
from .common import ROOT,MODEL,write


def activate():
    from ..dense_ar_bi0_shortanswer_pd import common
    common.ROOT=ROOT
    common.MODEL=MODEL
    common.write=write
