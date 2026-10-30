"""Reuse the finalized numerical correction without changing its source."""
import os
from .common import ROOT,VARIANTS


def install():
    if 'RCC_RUN' not in os.environ:return
    from ..looped_pd_b_numerical_diagnosis import common as numerical
    numerical.ROOT=ROOT
    os.environ['B_NUMERICAL_RUN']=os.environ['RCC_RUN']
    os.environ['B_NUMERICAL_VARIANT']=VARIANTS[os.environ['RCC_MODEL']]
    from ..looped_pd_b_numerical_diagnosis.patches import install as qualified_patch
    qualified_patch()
